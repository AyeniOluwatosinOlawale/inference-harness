from __future__ import annotations

import time

import anthropic

from ..types import RunResult, ScenarioDef, classify_bottleneck
from .base import BaseBackend

_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7":   {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "claude-haiku-4-5":  {"input": 1.00, "output":  5.00, "cache_write": 1.25, "cache_read": 0.10},
}

_MAX_TOKENS: dict[str, int] = {
    "short": 512, "medium": 1024, "long": 4096, "complex": 8192,
}


def _compute_cost(model: str, input_tok: int, output_tok: int,
                  cache_read: int, cache_write: int) -> float:
    p = _PRICING.get(model, _PRICING["claude-opus-4-7"])
    regular = max(0, input_tok - cache_read - cache_write)
    return (
        regular * p["input"] / 1_000_000
        + output_tok * p["output"] / 1_000_000
        + cache_read * p["cache_read"] / 1_000_000
        + cache_write * p["cache_write"] / 1_000_000
    )


class AnthropicBackend(BaseBackend):
    backend_name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        use_cache: bool = True,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.use_cache = use_cache
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )

    def _system_param(self, text: str) -> list[dict] | str:
        if not self.use_cache:
            return text
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    async def measure_single(self, scenario: ScenarioDef, *, run_index: int = 0) -> RunResult:
        max_tokens = _MAX_TOKENS.get(scenario.name, 1024)
        system = self._system_param(scenario.system)
        extra: dict = {}
        if scenario.use_thinking and self.model == "claude-opus-4-7":
            extra["thinking"] = {"type": "adaptive"}

        t0 = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        input_tok = output_tok = cache_read = cache_write = reasoning_tok = 0
        request_id: str | None = None

        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": scenario.user}],
                **extra,
            ) as stream:
                try:
                    request_id = stream.response.headers.get("request-id")
                except Exception:
                    pass

                async for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta and getattr(delta, "type", "") == "text_delta":
                                text = delta.text
                                if text:
                                    if ttft_ms is None:
                                        ttft_ms = (time.perf_counter() - t0) * 1000
                                    chunks.append(text)

                msg = await stream.get_final_message()
                total_ms = (time.perf_counter() - t0) * 1000
                u = msg.usage
                input_tok = u.input_tokens
                output_tok = u.output_tokens
                cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
                cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
                reasoning_tok = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0

        except Exception as exc:
            total_ms = (time.perf_counter() - t0) * 1000
            return RunResult(
                scenario=scenario.name, backend=self.backend_name, model=self.model,
                run_index=run_index, ttft_ms=0.0, total_ms=total_ms, tpot_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        if ttft_ms is None:
            ttft_ms = total_ms
        tpot_ms = (total_ms - ttft_ms) / max(output_tok - 1, 1)
        tps = output_tok / (total_ms / 1000) if total_ms > 0 else 0.0
        prefill_tps = input_tok / (ttft_ms / 1000) if ttft_ms > 0 else 0.0
        decode_tps = output_tok / ((total_ms - ttft_ms) / 1000) if (total_ms - ttft_ms) > 0 else 0.0
        cost = _compute_cost(self.model, input_tok, output_tok, cache_read, cache_write)
        bottleneck = classify_bottleneck(0.0, 0.0, ttft_ms, tpot_ms, output_tok)

        return RunResult(
            scenario=scenario.name, backend=self.backend_name, model=self.model,
            run_index=run_index,
            ttft_ms=ttft_ms, total_ms=total_ms, tpot_ms=tpot_ms,
            throughput_tps=tps,
            prefill_throughput_tps=prefill_tps,
            decode_throughput_tps=decode_tps,
            bottleneck=bottleneck,
            input_tokens=input_tok, output_tokens=output_tok,
            cache_read_tokens=cache_read, cache_creation_tokens=cache_write,
            reasoning_tokens=reasoning_tok,
            cost_usd=cost,
            response_text="".join(chunks),
            request_id=request_id,
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
