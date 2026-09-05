from __future__ import annotations

import time

import httpx
import openai

from ..types import RunResult, ScenarioDef, ServerMetrics, classify_bottleneck
from .base import BaseBackend

_MAX_TOKENS: dict[str, int] = {"short": 512, "medium": 1024, "long": 4096, "complex": 8192}


async def _scrape_trtllm_metrics(base_url: str, timeout: float = 2.0) -> ServerMetrics:
    """Parse TensorRT-LLM Prometheus /metrics endpoint."""
    sm = ServerMetrics()
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{base_url}/metrics")
            r.raise_for_status()
            for line in r.text.splitlines():
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                key, val = parts[0], parts[-1]
                try:
                    fval = float(val)
                except ValueError:
                    continue
                if "tensorrtllm_kv_cache_hit_rate" in key:
                    sm.kv_cache_hit_rate = fval
                elif "tensorrtllm_kv_cache_fraction_used" in key:
                    sm.kv_cache_utilization = fval
                elif "tensorrtllm_inflight_requests" in key:
                    sm.num_running_requests = int(fval)
                elif "tensorrtllm_request_queue_latency" in key:
                    sm.num_queued_requests = int(fval)
                # TRT-LLM 0.6+
                elif "tensorrtllm_first_token_latency" in key:
                    sm.prefill_ms = fval * 1000  # seconds → ms
                elif "tensorrtllm_time_per_output_token" in key:
                    sm.decode_ms = fval * 1000
    except Exception:
        pass
    return sm


class TRTLLMBackend(BaseBackend):
    backend_name = "tensorrt"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8001",
        api_key: str = "EMPTY",
        gpu_cost_per_hour: float = 0.0,
        collect_server_metrics: bool = True,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.collect_server_metrics = collect_server_metrics
        self._client = openai.AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=api_key,
            timeout=timeout,
        )

    async def measure_single(self, scenario: ScenarioDef, *, run_index: int = 0) -> RunResult:
        max_tokens = _MAX_TOKENS.get(scenario.name, 1024)
        messages = [
            {"role": "system", "content": scenario.system},
            {"role": "user",   "content": scenario.user},
        ]
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        input_tok = output_tok = 0

        try:
            # TRT-LLM older versions may not support stream_options
            try:
                stream = await self._client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens,
                    stream=True, stream_options={"include_usage": True},
                )
            except (openai.BadRequestError, openai.APIError):
                stream = await self._client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens, stream=True,
                )

            async for chunk in stream:
                content = chunk.choices[0].delta.content if chunk.choices else ""
                if content:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    chunks.append(content)
                if getattr(chunk, "usage", None):
                    input_tok = chunk.usage.prompt_tokens or 0
                    output_tok = chunk.usage.completion_tokens or 0

            total_ms = (time.perf_counter() - t0) * 1000

        except Exception as exc:
            total_ms = (time.perf_counter() - t0) * 1000
            return RunResult(
                scenario=scenario.name, backend=self.backend_name, model=self.model,
                run_index=run_index, ttft_ms=0.0, total_ms=total_ms, tpot_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        if ttft_ms is None:
            ttft_ms = total_ms
        if output_tok == 0:
            output_tok = max(1, len("".join(chunks)) // 4)

        tpot_ms = (total_ms - ttft_ms) / max(output_tok - 1, 1)
        tps = output_tok / (total_ms / 1000) if total_ms > 0 else 0.0
        cost = (total_ms / 1000 / 3600) * self.gpu_cost_per_hour

        sm = ServerMetrics()
        if self.collect_server_metrics:
            sm = await _scrape_trtllm_metrics(self.base_url)

        prefill_ms = sm.prefill_ms if sm.prefill_ms > 0 else 0.0
        decode_ms = (sm.decode_ms * output_tok) if sm.decode_ms > 0 else 0.0

        return RunResult(
            scenario=scenario.name, backend=self.backend_name, model=self.model,
            run_index=run_index,
            ttft_ms=ttft_ms, total_ms=total_ms, tpot_ms=tpot_ms,
            prefill_ms=prefill_ms, decode_ms=decode_ms,
            throughput_tps=tps,
            prefill_throughput_tps=input_tok / (ttft_ms / 1000) if ttft_ms > 0 and input_tok > 0 else 0.0,
            decode_throughput_tps=tps,
            bottleneck=classify_bottleneck(prefill_ms, decode_ms, ttft_ms, tpot_ms, output_tok),
            input_tokens=input_tok, output_tokens=output_tok,
            cost_usd=cost,
            kv_cache_hit_rate=sm.kv_cache_hit_rate,
            kv_cache_utilization=sm.kv_cache_utilization,
            response_text="".join(chunks),
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        return [self.model]

    async def close(self) -> None:
        await self._client.close()
