from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING

from .benchmarks.runner import ConcurrencyDataPoint
from .types import AggregatedMetrics, RunResult

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _ms(v: float) -> str:
    return f"{v:>8.1f}ms"

def _tps(v: float) -> str:
    return f"{v:>7.1f}"

def _cost(v: float) -> str:
    return f"${v:.5f}" if v > 0 else "   —   "

def _pct(v: float | None, invert: bool = False) -> str:
    if v is None:
        return "  —  "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"

def _bottleneck_label(b: str) -> str:
    labels = {
        "prefill": "⚠ PREFILL",
        "decode":  "⚠ DECODE",
        "balanced": "✓ BALANCED",
        "unknown": "  —",
    }
    return labels.get(b, b)


# ---------------------------------------------------------------------------
# Baseline latency + throughput + bottleneck table
# ---------------------------------------------------------------------------

def print_baseline_table(aggregated: list[AggregatedMetrics]) -> None:
    has_gpu = any(getattr(m, "gpu_memory_mean_mb", 0.0) > 0.0 for m in aggregated)
    has_prefill = any(m.prefill_p50_ms > 0 for m in aggregated)

    if _RICH:
        t = Table(title="Baseline: Latency, Throughput & Bottleneck", box=box.SIMPLE_HEAVY)
        t.add_column("Scenario", style="bold")
        t.add_column("Backend")
        t.add_column("N", justify="right")
        t.add_column("TTFT p50", justify="right")
        t.add_column("TTFT p95", justify="right")
        if has_prefill:
            t.add_column("Prefill p50", justify="right")
            t.add_column("Decode p50", justify="right")
        t.add_column("TPOT p50", justify="right")
        t.add_column("TPS (mean)", justify="right")
        t.add_column("Prefill TPS", justify="right")
        t.add_column("Decode TPS", justify="right")
        t.add_column("Cost/req", justify="right")
        t.add_column("Bottleneck", justify="center")
        if has_gpu:
            t.add_column("GPU Mem", justify="right")

        for m in aggregated:
            bn = m.dominant_bottleneck
            style = "red" if bn == "decode" else "yellow" if bn == "prefill" else "green"
            row = [
                m.scenario, m.backend, str(m.n),
                _ms(m.ttft_p50_ms), _ms(m.ttft_p95_ms),
            ]
            if has_prefill:
                row += [_ms(m.prefill_p50_ms), _ms(m.decode_p50_ms)]
            row += [
                _ms(m.tpot_p50_ms),
                _tps(m.throughput_mean_tps),
                _tps(m.prefill_throughput_mean_tps),
                _tps(m.decode_throughput_mean_tps),
                _cost(m.cost_mean_usd),
                _bottleneck_label(bn),
            ]
            if has_gpu:
                row.append(f"{getattr(m, 'gpu_memory_mean_mb', 0.0):.0f} MB")
            t.add_row(*row, style=style)
        _console.print(t)
    else:
        _plain_baseline(aggregated, has_prefill, has_gpu)


def _plain_baseline(aggregated: list[AggregatedMetrics], has_prefill: bool, has_gpu: bool) -> None:
    sep = "-" * 110
    print(f"\n{'BASELINE: LATENCY, THROUGHPUT & BOTTLENECK':^110}")
    print(sep)
    hdr = f"{'Scenario':<10} {'Backend':<12} {'N':>3} {'TTFT p50':>10} {'TTFT p95':>10}"
    if has_prefill:
        hdr += f" {'Prefill p50':>12} {'Decode p50':>12}"
    hdr += f" {'TPOT p50':>10} {'TPS':>8} {'Pre TPS':>8} {'Dec TPS':>8} {'Cost/req':>10} {'Bottleneck':<14}"
    if has_gpu:
        hdr += f" {'GPU Mem':>10}"
    print(hdr)
    print(sep)
    for m in aggregated:
        row = (
            f"{m.scenario:<10} {m.backend:<12} {m.n:>3} "
            f"{_ms(m.ttft_p50_ms):>10} {_ms(m.ttft_p95_ms):>10}"
        )
        if has_prefill:
            row += f" {_ms(m.prefill_p50_ms):>12} {_ms(m.decode_p50_ms):>12}"
        row += (
            f" {_ms(m.tpot_p50_ms):>10}"
            f" {_tps(m.throughput_mean_tps):>8}"
            f" {_tps(m.prefill_throughput_mean_tps):>8}"
            f" {_tps(m.decode_throughput_mean_tps):>8}"
            f" {_cost(m.cost_mean_usd):>10}"
            f" {_bottleneck_label(m.dominant_bottleneck):<14}"
        )
        if has_gpu:
            row += f" {getattr(m, 'gpu_memory_mean_mb', 0.0):>8.0f} MB"
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Bottleneck summary with recommendations
# ---------------------------------------------------------------------------

def print_bottleneck_summary(aggregated: list[AggregatedMetrics]) -> None:
    print("\n" + "=" * 80)
    print("  BOTTLENECK ANALYSIS")
    print("=" * 80)

    prefill_bound = [m for m in aggregated if m.dominant_bottleneck == "prefill"]
    decode_bound  = [m for m in aggregated if m.dominant_bottleneck == "decode"]
    balanced      = [m for m in aggregated if m.dominant_bottleneck == "balanced"]

    for m in aggregated:
        pf = _ms(m.prefill_p50_ms) if m.prefill_p50_ms > 0 else f"{_ms(m.ttft_p50_ms)} (TTFT proxy)"
        dc = _ms(m.decode_p50_ms)  if m.decode_p50_ms  > 0 else f"{_ms(m.tpot_p50_ms)}×tok (proxy)"
        print(f"  {m.scenario:<10}  Prefill: {pf}  Decode: {dc}  → {_bottleneck_label(m.dominant_bottleneck)}")

    print()
    if decode_bound:
        names = ", ".join(m.scenario for m in decode_bound)
        print(f"  DECODE-bound ({names}):")
        print("    → Enable speculative decoding (draft model)")
        print("    → Increase batch size to amortize HBM reads")
        print("    → Quantize weights (INT4/FP8) to reduce memory bandwidth")
        print("    → Enable PagedAttention / prefix caching")
    if prefill_bound:
        names = ", ".join(m.scenario for m in prefill_bound)
        print(f"  PREFILL-bound ({names}):")
        print("    → Enable chunked prefill to reduce TTFT")
        print("    → Increase tensor parallelism degree")
        print("    → Use prefix caching for shared system prompts")
        print("    → Consider FlashAttention-3 / FlashInfer for faster context processing")
    if balanced:
        names = ", ".join(m.scenario for m in balanced)
        print(f"  BALANCED ({names}) — well-tuned, no single dominant bottleneck")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Concurrency sweep table
# ---------------------------------------------------------------------------

def print_concurrency_table(points: list[ConcurrencyDataPoint]) -> None:
    if _RICH:
        t = Table(title="Concurrency Sweep", box=box.SIMPLE_HEAVY)
        for col in ["Concurrency", "TTFT p50", "TTFT p95", "TPS (mean)", "Prefill TPS", "Decode TPS", "Bottleneck", "Errors"]:
            t.add_column(col, justify="right" if col not in ("Bottleneck",) else "center")
        for p in points:
            t.add_row(
                str(p.concurrency), _ms(p.ttft_p50_ms), _ms(p.ttft_p95_ms),
                _tps(p.throughput_mean_tps), _tps(p.prefill_throughput_mean_tps),
                _tps(p.decode_throughput_mean_tps),
                _bottleneck_label(p.dominant_bottleneck), str(p.errors),
            )
        _console.print(t)
    else:
        print("\nCONCURRENCY SWEEP")
        for p in points:
            print(
                f"  c={p.concurrency:>2}  TTFT p50={_ms(p.ttft_p50_ms)} p95={_ms(p.ttft_p95_ms)}"
                f"  TPS={_tps(p.throughput_mean_tps)}"
                f"  {_bottleneck_label(p.dominant_bottleneck)}  errors={p.errors}"
            )


# ---------------------------------------------------------------------------
# Cross-backend comparison
# ---------------------------------------------------------------------------

def print_compare_table(
    closed: list[AggregatedMetrics],
    open_m: list[AggregatedMetrics],
    closed_label: str = "Closed API",
    open_label: str = "Open Model",
) -> None:
    closed_map = {m.scenario: m for m in closed}
    open_map   = {m.scenario: m for m in open_m}
    scenarios  = sorted(set(closed_map) | set(open_map))

    print("\n" + "=" * 110)
    print(f"  COMPARISON: {closed_label} vs {open_label}")
    print("=" * 110)
    hdr = (
        f"{'Scenario':<10} "
        f"{'TTFT p50 (A)':>14} {'TTFT p50 (B)':>14} {'Δ TTFT':>8}  "
        f"{'TPS (A)':>8} {'TPS (B)':>8} {'Δ TPS':>8}  "
        f"{'Cost (A)':>10} {'Cost (B)':>10} {'Δ Cost':>8}"
    )
    print(hdr)
    print("-" * 110)

    for s in scenarios:
        c = closed_map.get(s)
        o = open_map.get(s)
        ttft_delta = ((1 - o.ttft_p50_ms / c.ttft_p50_ms) * 100) if c and o and c.ttft_p50_ms > 0 else None
        tps_delta  = ((o.throughput_mean_tps / c.throughput_mean_tps - 1) * 100) if c and o and c.throughput_mean_tps > 0 else None
        cost_delta = ((1 - o.cost_mean_usd / c.cost_mean_usd) * 100) if c and o and c.cost_mean_usd > 0 else None
        print(
            f"{s:<10} "
            f"{(_ms(c.ttft_p50_ms) if c else '      —'):>14} "
            f"{(_ms(o.ttft_p50_ms) if o else '      —'):>14} "
            f"{_pct(ttft_delta):>8}  "
            f"{(_tps(c.throughput_mean_tps) if c else '  —'):>8} "
            f"{(_tps(o.throughput_mean_tps) if o else '  —'):>8} "
            f"{_pct(tps_delta):>8}  "
            f"{(_cost(c.cost_mean_usd) if c else '  —'):>10} "
            f"{(_cost(o.cost_mean_usd) if o else '  —'):>10} "
            f"{_pct(cost_delta):>8}"
        )
    print("=" * 110)
    print(f"  Δ sign: positive TTFT/Cost = {open_label} is faster/cheaper. Positive TPS = higher throughput.")


# ---------------------------------------------------------------------------
# JSON / CSV export
# ---------------------------------------------------------------------------

def export_json(
    path: str,
    all_results: list[RunResult],
    aggregated: list[AggregatedMetrics],
    metadata: dict | None = None,
) -> None:
    payload = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **(metadata or {}),
        },
        "aggregated": [asdict(m) for m in aggregated],
        "runs": [asdict(r) for r in all_results],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Results saved → {path}")


def export_csv(path: str, aggregated: list[AggregatedMetrics]) -> None:
    if not aggregated:
        return
    rows = [asdict(m) for m in aggregated]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved → {path}")
