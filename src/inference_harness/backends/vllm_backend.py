from __future__ import annotations

import time

import httpx
import openai

from ..types import RunResult, ScenarioDef, ServerMetrics, classify_bottleneck
from .base import BaseBackend

_MAX_TOKENS: dict[str, int] = {"short": 512, "medium": 1024, "long": 4096, "complex": 8192}


async def _scrape_vllm_metrics(base_url: str, timeout: float = 2.0) -> ServerMetrics:
    """Parse vLLM Prometheus /metrics endpoint."""
    sm = ServerMetrics()
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{base_url}/metrics")
            r.raise_for_status()
            for line in r.text.splitlines():
                if line.startswith("#"):
                    continue
                if "vllm:cache_hit_rate" in line and "{" not in line.split()[0]:
                    sm.kv_cache_hit_rate = float(line.split()[-1])
                elif "vllm:gpu_cache_usage_perc" in line and "{" not in line.split()[0]:
                    sm.kv_cache_utilization = float(line.split()[-1])
                elif "vllm:num_requests_running" in line:
                    sm.num_running_requests = int(float(line.split()[-1]))
                elif "vllm:num_requests_waiting" in line:
                    sm.num_queued_requests = int(float(line.split()[-1]))
                elif "vllm:time_to_first_token_seconds_sum" in line:
                    pass  # histogram — skip raw sum
                elif "vllm:time_per_output_token_seconds_sum" in line:
                    pass
    except Exception:
        pass
    return sm


class VLLMBackend(BaseBackend):
    backend_name = "vllm"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000",
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
            try:
                stream = await self._client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens,
                    stream=True, stream_options={"include_usage": True},
                )
            except openai.BadRequestError as e:
                if "stream_options" not in str(e):
                    raise
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
        prefill_tps = input_tok / (ttft_ms / 1000) if ttft_ms > 0 and input_tok > 0 else 0.0
        decode_tps = output_tok / ((total_ms - ttft_ms) / 1000) if (total_ms - ttft_ms) > 0 else 0.0
        cost = (total_ms / 1000 / 3600) * self.gpu_cost_per_hour

        # Collect server-side metrics after the request
        sm = ServerMetrics()
        if self.collect_server_metrics:
            sm = await _scrape_vllm_metrics(self.base_url)

        # Use server metrics to get explicit prefill/decode where available
        prefill_ms = sm.prefill_ms if sm.prefill_ms > 0 else 0.0
        decode_ms = sm.decode_ms if sm.decode_ms > 0 else 0.0

        return RunResult(
            scenario=scenario.name, backend=self.backend_name, model=self.model,
            run_index=run_index,
            ttft_ms=ttft_ms, total_ms=total_ms, tpot_ms=tpot_ms,
            prefill_ms=prefill_ms, decode_ms=decode_ms,
            throughput_tps=tps,
            prefill_throughput_tps=prefill_tps if prefill_tps > 0 else sm.prefill_throughput_tps,
            decode_throughput_tps=decode_tps if decode_tps > 0 else sm.decode_throughput_tps,
            bottleneck=classify_bottleneck(prefill_ms, decode_ms, ttft_ms, tpot_ms, output_tok),
            queue_time_ms=0.0,
            input_tokens=input_tok, output_tokens=output_tok,
            cost_usd=cost,
            kv_cache_hit_rate=sm.kv_cache_hit_rate,
            kv_cache_utilization=sm.kv_cache_utilization,
            response_text="".join(chunks),
        )

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            page = await self._client.models.list()
            return [m.id for m in page.data]
        except Exception:
            return [self.model]

    async def close(self) -> None:
        await self._client.close()
