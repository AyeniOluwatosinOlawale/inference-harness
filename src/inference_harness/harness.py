from __future__ import annotations

import asyncio
import os
from typing import Literal

from .backends import get_backend
from .backends.base import BaseBackend
from .benchmarks.runner import (
    ConcurrencyDataPoint,
    run_all_scenarios,
    run_concurrency_sweep,
    run_scenario,
)
from .profiling.system_profiler import SystemProfiler
from .types import AggregatedMetrics, RunResult, ScenarioDef, aggregate


class Harness:
    """
    Main library entry point for inference-harness.

    Usage:
        h = Harness("anthropic", "claude-sonnet-4-6")
        results = await h.run(ScenarioDef.medium(), runs=5)
        print(h.aggregate(results))

        # Compare two backends
        open_h = Harness("vllm", "meta-llama/Llama-3.1-8B", base_url="http://localhost:8000")
        comparison = await h.compare(open_h, ScenarioDef.medium())
    """

    def __init__(
        self,
        backend: str,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "",
        profiling: Literal["none", "system", "cuda", "both"] = "system",
        gpu_cost_per_hour: float = 0.0,
    ) -> None:
        self.backend_name = backend
        self.model = model
        self.profiling = profiling
        self._backend: BaseBackend = get_backend(
            backend, model,
            api_key=api_key or _env_key(backend),
            base_url=base_url,
            gpu_cost_per_hour=gpu_cost_per_hour,
        )
        self._profiler = SystemProfiler() if profiling in ("system", "both") else None

    async def run(
        self,
        scenario: ScenarioDef,
        runs: int = 3,
    ) -> list[RunResult]:
        return await run_scenario(self._backend, scenario, runs=runs)

    async def run_all(
        self,
        scenarios: list[ScenarioDef] | None = None,
        runs: int = 3,
    ) -> tuple[list[RunResult], list[AggregatedMetrics]]:
        s = scenarios or ScenarioDef.all()
        return await run_all_scenarios(self._backend, s, runs=runs)

    async def sweep_concurrency(
        self,
        scenario: ScenarioDef,
        levels: list[int] | None = None,
        runs_per_level: int = 3,
    ) -> list[ConcurrencyDataPoint]:
        lvls = levels or [1, 2, 4, 8, 16]
        return await run_concurrency_sweep(self._backend, scenario, lvls, runs_per_level)

    async def compare(
        self,
        other: "Harness",
        scenario: ScenarioDef,
        runs: int = 3,
    ) -> tuple[list[RunResult], list[RunResult]]:
        """Run same scenario on both backends concurrently and return both result lists."""
        self_results, other_results = await asyncio.gather(
            self.run(scenario, runs=runs),
            other.run(scenario, runs=runs),
        )
        return self_results, other_results

    def aggregate(self, results: list[RunResult]) -> AggregatedMetrics:
        return aggregate(results)

    async def health_check(self) -> bool:
        return await self._backend.health_check()

    async def close(self) -> None:
        await self._backend.close()

    async def __aenter__(self) -> "Harness":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()


def _env_key(backend: str) -> str | None:
    keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "gemini":    "GOOGLE_API_KEY",
        "groq":      "GROQ_API_KEY",
        "together":  "TOGETHER_API_KEY",
    }
    env = keys.get(backend.lower())
    return os.getenv(env) if env else None
