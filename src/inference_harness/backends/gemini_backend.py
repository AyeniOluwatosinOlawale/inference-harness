from __future__ import annotations

import time

from ..types import RunResult, ScenarioDef, classify_bottleneck
from .base import BaseBackend

_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash":        {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-lite":   {"input": 0.0375, "output": 0.15},
    "gemini-1.5-pro":          {"input": 1.25,  "output": 5.00},
    "gemini-1.5-flash":        {"input": 0.075, "output": 0.30},
}

_MAX_TOKENS: dict[str, int] = {"short": 512, "medium": 1024, "long": 4096, "complex": 8192}


class GeminiBackend(BaseBackend):
    backend_name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            if self._api_key:
                genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=None,
            )
        return self._client

    async def measure_single(self, scenario: ScenarioDef, *, run_index: int = 0) -> RunResult:
        import google.generativeai as genai
        if self._api_key:
            genai.configure(api_key=self._api_key)

        max_tokens = _MAX_TOKENS.get(scenario.name, 1024)
        client = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=scenario.system,
        )

        t0 = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        input_tok = output_tok = 0

        try:
            response = await client.generate_content_async(
                scenario.user,
                generation_config={"max_output_tokens": max_tokens},
                stream=True,
            )
            async for chunk in response:
                text = chunk.text if hasattr(chunk, "text") else ""
                if text:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    chunks.append(text)

            total_ms = (time.perf_counter() - t0) * 1000
            # Usage metadata on the final resolved response
            try:
                resolved = await response.resolve()
                meta = resolved.usage_metadata
                input_tok = meta.prompt_token_count or 0
                output_tok = meta.candidates_token_count or 0
            except Exception:
                output_tok = max(1, len("".join(chunks)) // 4)

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
        p = _PRICING.get(self.model, {"input": 0.075, "output": 0.30})
        cost = input_tok * p["input"] / 1_000_000 + output_tok * p["output"] / 1_000_000

        return RunResult(
            scenario=scenario.name, backend=self.backend_name, model=self.model,
            run_index=run_index,
            ttft_ms=ttft_ms, total_ms=total_ms, tpot_ms=tpot_ms,
            throughput_tps=tps,
            prefill_throughput_tps=input_tok / (ttft_ms / 1000) if ttft_ms > 0 else 0.0,
            decode_throughput_tps=output_tok / ((total_ms - ttft_ms) / 1000) if (total_ms - ttft_ms) > 0 else 0.0,
            bottleneck=classify_bottleneck(0.0, 0.0, ttft_ms, tpot_ms, output_tok),
            input_tokens=input_tok, output_tokens=output_tok,
            cost_usd=cost,
            response_text="".join(chunks),
        )

    async def health_check(self) -> bool:
        try:
            import google.generativeai as genai
            if self._api_key:
                genai.configure(api_key=self._api_key)
            list(genai.list_models())
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        return list(_PRICING.keys())

    async def close(self) -> None:
        pass
