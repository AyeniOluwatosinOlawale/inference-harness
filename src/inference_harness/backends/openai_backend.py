from __future__ import annotations

import time

import openai

from ..types import RunResult, ScenarioDef, classify_bottleneck
from .base import BaseBackend

_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":          {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":     {"input": 0.15, "output":  0.60},
    "o1":              {"input": 15.0, "output":  60.0},
    "o1-mini":         {"input": 3.00, "output":  12.0},
    "o3-mini":         {"input": 1.10, "output":   4.4},
}

_MAX_TOKENS: dict[str, int] = {"short": 512, "medium": 1024, "long": 4096, "complex": 8192}


def _compute_cost(model: str, input_tok: int, output_tok: int) -> float:
    p = _PRICING.get(model, {"input": 2.50, "output": 10.0})
    return input_tok * p["input"] / 1_000_000 + output_tok * p["output"] / 1_000_000


class OpenAIBackend(BaseBackend):
    backend_name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
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
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                content = ""
                if chunk.choices:
                    content = chunk.choices[0].delta.content or ""
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
        prefill_tps = input_tok / (ttft_ms / 1000) if ttft_ms > 0 else 0.0
        decode_tps = output_tok / ((total_ms - ttft_ms) / 1000) if (total_ms - ttft_ms) > 0 else 0.0

        return RunResult(
            scenario=scenario.name, backend=self.backend_name, model=self.model,
            run_index=run_index,
            ttft_ms=ttft_ms, total_ms=total_ms, tpot_ms=tpot_ms,
            throughput_tps=tps, prefill_throughput_tps=prefill_tps, decode_throughput_tps=decode_tps,
            bottleneck=classify_bottleneck(0.0, 0.0, ttft_ms, tpot_ms, output_tok),
            input_tokens=input_tok, output_tokens=output_tok,
            cost_usd=_compute_cost(self.model, input_tok, output_tok),
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
            return list(_PRICING.keys())

    async def close(self) -> None:
        await self._client.close()
