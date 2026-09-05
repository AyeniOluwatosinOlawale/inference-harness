from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..backends.base import BaseBackend
from ..types import AggregatedMetrics, RunResult, ScenarioDef, aggregate


@dataclass
class ConcurrencyDataPoint:
    concurrency: int
    scenario: str
    ttft_p50_ms: float
    ttft_p95_ms: float
    throughput_mean_tps: float
    prefill_throughput_mean_tps: float
    decode_throughput_mean_tps: float
    dominant_bottleneck: str
    errors: int


async def run_scenario(
    backend: BaseBackend,
    scenario: ScenarioDef,
    runs: int = 3,
) -> list[RunResult]:
    results: list[RunResult] = []
    for i in range(runs):
        r = await backend.measure_single(scenario, run_index=i)
        results.append(r)
    return results


async def run_all_scenarios(
    backend: BaseBackend,
    scenarios: list[ScenarioDef],
    runs: int = 3,
) -> tuple[list[RunResult], list[AggregatedMetrics]]:
    all_results: list[RunResult] = []
    aggregated: list[AggregatedMetrics] = []
    for scenario in scenarios:
        results = await run_scenario(backend, scenario, runs)
        all_results.extend(results)
        try:
            aggregated.append(aggregate(results))
        except ValueError:
            pass
    return all_results, aggregated


async def run_concurrency_sweep(
    backend: BaseBackend,
    scenario: ScenarioDef,
    levels: list[int],
    runs_per_level: int = 3,
) -> list[ConcurrencyDataPoint]:
    """
    For each concurrency level, fire N concurrent requests and measure
    TTFT degradation and throughput scaling.
    """
    points: list[ConcurrencyDataPoint] = []
    for c in levels:
        tasks = [
            backend.measure_single(scenario, run_index=i)
            for i in range(c * runs_per_level)
        ]
        # Send `c` requests at a time
        all_results: list[RunResult] = []
        for batch_start in range(0, len(tasks), c):
            batch = tasks[batch_start:batch_start + c]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for r in batch_results:
                if isinstance(r, RunResult):
                    all_results.append(r)

        good = [r for r in all_results if r.succeeded]
        errors = len(all_results) - len(good)

        if not good:
            points.append(ConcurrencyDataPoint(
                concurrency=c, scenario=scenario.name,
                ttft_p50_ms=0.0, ttft_p95_ms=0.0,
                throughput_mean_tps=0.0,
                prefill_throughput_mean_tps=0.0,
                decode_throughput_mean_tps=0.0,
                dominant_bottleneck="unknown", errors=errors,
            ))
            continue

        try:
            agg = aggregate(good)
            points.append(ConcurrencyDataPoint(
                concurrency=c, scenario=scenario.name,
                ttft_p50_ms=agg.ttft_p50_ms,
                ttft_p95_ms=agg.ttft_p95_ms,
                throughput_mean_tps=agg.throughput_mean_tps,
                prefill_throughput_mean_tps=agg.prefill_throughput_mean_tps,
                decode_throughput_mean_tps=agg.decode_throughput_mean_tps,
                dominant_bottleneck=agg.dominant_bottleneck,
                errors=errors,
            ))
        except ValueError:
            pass

    return points
