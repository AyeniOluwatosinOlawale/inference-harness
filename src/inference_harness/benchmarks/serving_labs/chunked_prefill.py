"""
Phase 7 — Lab 2: Chunked Prefill
=================================
Measures the TTFT vs throughput tradeoff introduced by chunked prefill.

Without chunked prefill: long prompts block the GPU for the entire prefill
duration, starving concurrent decode steps and causing TTFT spikes.

With chunked prefill (--enable-chunked-prefill): long prompts are split into
fixed-size chunks, interleaved with ongoing decode work. TTFT for short
concurrent requests improves dramatically; throughput slightly decreases.

Key parameters:
  --max-num-batched-tokens   Controls chunk size (default: 512 in vLLM)
  --enable-chunked-prefill   Enables the feature

Setup:
  # Baseline (no chunked prefill)
  python -m vllm.entrypoints.openai.api_server \\
    --model <model> --port 8000 --disable-chunked-prefill

  # With chunked prefill
  python -m vllm.entrypoints.openai.api_server \\
    --model <model> --port 8001 \\
    --enable-chunked-prefill --max-num-batched-tokens 512

Usage:
  python -m inference_harness.benchmarks.serving_labs.chunked_prefill \\
    --baseline-url http://localhost:8000 \\
    --chunked-url  http://localhost:8001 \\
    --model <model>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import openai


# Workload: mix of requests with long prompts (prefill-heavy) and short prompts
# submitted concurrently. The short requests measure TTFT degradation when
# a long prefill is monopolising the GPU.

LONG_PROMPT = (
    "You are an expert software engineer. Analyse the following large codebase "
    "and provide a comprehensive review covering architecture, performance, security, "
    "and maintainability. Here is the code: "
    + "def process(items):\n    for item in items:\n        yield item * 2\n" * 80
)

SHORT_PROMPTS = [
    "What is 2 + 2?",
    "Name a colour.",
    "Say hello.",
    "What day comes after Monday?",
    "Name a country in Europe.",
]


async def timed_request(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 50,
) -> dict:
    t0 = time.perf_counter()
    first_token = None
    output_tokens = 0

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        temperature=0.0,
    )
    async for chunk in stream:
        if first_token is None and chunk.choices and chunk.choices[0].delta.content:
            first_token = time.perf_counter()
        if chunk.choices and chunk.choices[0].delta.content:
            output_tokens += 1

    completed = time.perf_counter()
    ttft_ms = ((first_token or completed) - t0) * 1000
    total_ms = (completed - t0) * 1000
    tpot_ms = (total_ms - ttft_ms) / max(output_tokens - 1, 1)

    return {
        "ttft_ms": round(ttft_ms, 1),
        "total_ms": round(total_ms, 1),
        "tpot_ms": round(tpot_ms, 1),
        "output_tokens": output_tokens,
    }


async def run_mixed_workload(
    url: str,
    model: str,
    n_long: int,
    n_short: int,
    concurrency: int,
) -> dict:
    """
    Submit long and short requests concurrently.
    The key insight: with chunked prefill, short requests complete quickly
    even while long prefills are being processed in chunks.
    """
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)

    long_results: list[dict] = []
    short_results: list[dict] = []

    async def send_long():
        async with sem:
            r = await timed_request(client, model, LONG_PROMPT, max_tokens=100)
            long_results.append(r)

    async def send_short(p: str):
        async with sem:
            r = await timed_request(client, model, p, max_tokens=20)
            short_results.append(r)

    long_tasks = [send_long() for _ in range(n_long)]
    short_tasks = [send_short(SHORT_PROMPTS[i % len(SHORT_PROMPTS)]) for i in range(n_short)]

    # Submit all concurrently — short and long requests compete for GPU
    await asyncio.gather(*long_tasks, *short_tasks)
    await client.close()

    def stats(results: list[dict], key: str) -> dict:
        vals = [r[key] for r in results]
        if not vals:
            return {"mean": 0, "p50": 0, "p95": 0}
        s = sorted(vals)
        return {
            "mean": round(statistics.mean(vals), 1),
            "p50": round(s[len(s) // 2], 1),
            "p95": round(s[int(len(s) * 0.95)], 1),
        }

    return {
        "short_ttft": stats(short_results, "ttft_ms"),
        "long_ttft": stats(long_results, "ttft_ms"),
        "short_tpot": stats(short_results, "tpot_ms"),
        "long_tpot": stats(long_results, "tpot_ms"),
        "short_n": len(short_results),
        "long_n": len(long_results),
    }


async def run_sweep(
    url: str,
    model: str,
    chunk_sizes: list[int | None],
    n_long: int,
    n_short: int,
    concurrency: int,
) -> list[dict]:
    """
    Sweep across chunk sizes by running the same workload multiple times.
    chunk_size=None means the server uses its configured value.
    """
    results = []
    for chunk_size in chunk_sizes:
        label = f"chunk={chunk_size}" if chunk_size else "server-default"
        result = await run_mixed_workload(url, model, n_long, n_short, concurrency)
        result["label"] = label
        results.append(result)
    return results


async def main(args: argparse.Namespace):
    print("=== Phase 7 Lab 2: Chunked Prefill ===\n")
    print(f"Long prompts:  {args.n_long}  (prefill-heavy, ~600 tokens)")
    print(f"Short prompts: {args.n_short} (decode-heavy, ~5 tokens)")
    print(f"Concurrency:   {args.concurrency}\n")

    print("--- Baseline (chunked prefill disabled) ---")
    baseline = await run_mixed_workload(
        args.baseline_url, args.model, args.n_long, args.n_short, args.concurrency
    )

    print("--- Chunked prefill enabled ---")
    chunked = await run_mixed_workload(
        args.chunked_url, args.model, args.n_long, args.n_short, args.concurrency
    )

    print("\n--- Results ---")
    print(f"\n{'Metric':>30} {'Baseline':>12} {'Chunked':>12} {'Delta':>10}")
    print("-" * 68)

    comparisons = [
        ("Short TTFT mean (ms)",  baseline["short_ttft"]["mean"],  chunked["short_ttft"]["mean"]),
        ("Short TTFT p95  (ms)",  baseline["short_ttft"]["p95"],   chunked["short_ttft"]["p95"]),
        ("Short TTFT p50  (ms)",  baseline["short_ttft"]["p50"],   chunked["short_ttft"]["p50"]),
        ("Long  TTFT mean (ms)",  baseline["long_ttft"]["mean"],   chunked["long_ttft"]["mean"]),
        ("Short TPOT mean (ms)",  baseline["short_tpot"]["mean"],  chunked["short_tpot"]["mean"]),
    ]

    for name, b, c in comparisons:
        if b > 0:
            delta = (c - b) / b * 100
            sign = "▼" if delta < 0 else "▲"
            better = "better" if (delta < 0 and "TTFT" in name) or (delta < 0 and "TPOT" in name) else ""
            print(f"{name:>30}  {b:>10.1f}  {c:>10.1f}  {sign}{abs(delta):.1f}% {better}")
        else:
            print(f"{name:>30}  {'N/A':>10}  {c:>10.1f}")

    short_ttft_improvement = (baseline["short_ttft"]["mean"] - chunked["short_ttft"]["mean"]) / baseline["short_ttft"]["mean"] * 100
    print(f"\nConclusion: short request TTFT {'improved' if short_ttft_improvement > 0 else 'worsened'} by {abs(short_ttft_improvement):.1f}% with chunked prefill")
    print("Trade-off: long request TTFT increases (chunked = more prefill iterations) but short requests stop waiting")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"baseline": baseline, "chunked": chunked}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 Chunked Prefill benchmark")
    parser.add_argument("--baseline-url", default="http://localhost:8000")
    parser.add_argument("--chunked-url", default="http://localhost:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-long", type=int, default=4)
    parser.add_argument("--n-short", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output")
    asyncio.run(main(parser.parse_args()))
