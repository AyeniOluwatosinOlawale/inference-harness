from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from typing import Optional

import typer

from .backends import get_backend
from .benchmarks.runner import run_all_scenarios, run_concurrency_sweep
from .harness import Harness
from .report import (
    export_csv,
    export_json,
    print_baseline_table,
    print_bottleneck_summary,
    print_compare_table,
    print_concurrency_table,
)
from .types import AggregatedMetrics, RunResult, ScenarioDef, aggregate

app = typer.Typer(name="inference-harness", help="Unified inference engineering harness.")


def _resolve_api_key(backend: str, provided: Optional[str]) -> Optional[str]:
    if provided:
        return provided
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
    }
    return os.getenv(env_map.get(backend.lower(), ""))


@app.command("run")
def cmd_run(
    backend: str = typer.Option(..., "--backend", "-b", help="anthropic | openai | gemini | groq | together | vllm | sglang | tensorrt"),
    model: str = typer.Option(..., "--model", "-m", help="Model name / ID"),
    scenario: Optional[str] = typer.Option(None, "--scenario", "-s", help="short | medium | long | complex (default: all)"),
    runs: int = typer.Option(3, "--runs", "-r", help="Runs per scenario"),
    base_url: str = typer.Option("", "--base-url", help="Base URL for open model servers"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key (overrides env var)"),
    gpu_cost: float = typer.Option(0.0, "--gpu-cost-per-hour", help="GPU cost USD/hr for cost tracking"),
    concurrency: bool = typer.Option(False, "--concurrency", help="Run concurrency sweep"),
    conc_levels: str = typer.Option("1,2,4,8,16", "--conc-levels", help="Comma-separated concurrency levels"),
    output_json: Optional[str] = typer.Option(None, "--output", "-o", help="Save results to JSON"),
    output_csv: Optional[str] = typer.Option(None, "--csv", help="Save aggregated metrics to CSV"),
    no_gpu_metrics: bool = typer.Option(False, "--no-gpu-metrics", help="Skip GPU profiling"),
    profile: str = typer.Option("system", "--profile", help="none | system | cuda | both"),
) -> None:
    """Run inference benchmark on a single backend."""
    key = _resolve_api_key(backend, api_key)
    scenarios = (
        [getattr(ScenarioDef, scenario)()] if scenario else ScenarioDef.all()
    )
    levels = [int(x) for x in conc_levels.split(",")]

    async def _main() -> None:
        b = get_backend(backend, model, api_key=key, base_url=base_url, gpu_cost_per_hour=gpu_cost)

        typer.echo(f"\n  Backend : {backend} / {model}")
        typer.echo(f"  Scenarios: {[s.name for s in scenarios]}  runs={runs}")

        all_results, aggregated = await run_all_scenarios(b, scenarios, runs=runs)

        print_baseline_table(aggregated)
        print_bottleneck_summary(aggregated)

        if concurrency:
            typer.echo("\n  Running concurrency sweep…")
            points = await run_concurrency_sweep(b, ScenarioDef.medium(), levels)
            print_concurrency_table(points)

        if output_json:
            export_json(output_json, all_results, aggregated, {"backend": backend, "model": model})
        if output_csv:
            export_csv(output_csv, aggregated)

        await b.close()

    asyncio.run(_main())


@app.command("compare")
def cmd_compare(
    closed: str = typer.Option(..., "--closed", help="Closed model backend name"),
    closed_model: str = typer.Option(..., "--closed-model", help="Closed model ID"),
    open_backend: str = typer.Option(..., "--open", help="Open model backend: vllm | sglang | tensorrt"),
    open_model: str = typer.Option(..., "--open-model", help="Open model name"),
    open_url: str = typer.Option("http://localhost:8000", "--open-url", help="Open model server URL"),
    scenario: Optional[str] = typer.Option(None, "--scenario", "-s"),
    runs: int = typer.Option(3, "--runs", "-r"),
    closed_api_key: Optional[str] = typer.Option(None, "--closed-key"),
    output_json: Optional[str] = typer.Option(None, "--output", "-o"),
) -> None:
    """Run the same scenarios on two backends and show side-by-side comparison."""
    scenarios = ([getattr(ScenarioDef, scenario)()] if scenario else ScenarioDef.all())

    async def _main() -> None:
        closed_key = _resolve_api_key(closed, closed_api_key)
        closed_b = get_backend(closed, closed_model, api_key=closed_key)
        open_b = get_backend(open_backend, open_model, base_url=open_url)

        typer.echo(f"\n  [A] {closed} / {closed_model}")
        typer.echo(f"  [B] {open_backend} / {open_model} @ {open_url}")

        (closed_results, closed_agg), (open_results, open_agg) = await asyncio.gather(
            run_all_scenarios(closed_b, scenarios, runs=runs),
            run_all_scenarios(open_b, scenarios, runs=runs),
        )

        print_baseline_table(closed_agg)
        print_baseline_table(open_agg)
        print_compare_table(
            closed_agg, open_agg,
            closed_label=f"{closed}/{closed_model}",
            open_label=f"{open_backend}/{open_model}",
        )
        print_bottleneck_summary(closed_agg + open_agg)

        if output_json:
            export_json(output_json, closed_results + open_results, closed_agg + open_agg)

        await closed_b.close()
        await open_b.close()

    asyncio.run(_main())


@app.command("dashboard")
def cmd_dashboard(
    port: int = typer.Option(8080, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host"),
) -> None:
    """Start the web dashboard (FastAPI + WebSocket)."""
    try:
        import uvicorn
        from .api.server import build_app
        uvicorn.run(build_app(), host=host, port=port, log_level="info")
    except ImportError:
        typer.echo("fastapi and uvicorn are required for the dashboard.", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
