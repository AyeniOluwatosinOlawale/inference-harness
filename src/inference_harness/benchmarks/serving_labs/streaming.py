"""
Phase 7 — Lab 4: Streaming Inference
======================================
Measures the latency difference between streaming and non-streaming responses,
and profiles the per-token delivery pattern (inter-token delay).

Streaming (SSE) is essential for:
  - Interactive chat UIs (perceived latency is TTFT, not total latency)
  - Long-running generations where the user reads as text arrives
  - Agents that process partial responses

Non-streaming waits for the full response — useful for batch processing
where total latency matters and perceived latency does not.

Metrics:
  - TTFT: time to first token (only meaningful in streaming mode)
  - TPOT: time per output token (inter-token delay)
  - Total latency: time to last token
  - Streaming vs non-streaming total latency comparison

Setup:
  python -m vllm.entrypoints.openai.api_server --model <model> --port 8000
  # or
  python -m sglang.launch_server --model <model> --port 30000

Usage:
  python -m inference_harness.benchmarks.serving_labs.streaming \\
    --url http://localhost:8000 --model <model>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import openai


PROMPTS = {
    "short":  ("What is the capital of France?", 30),
    "medium": ("Explain how gradient descent works in machine learning.", 150),
    "long":   ("Write a comprehensive guide to building scalable REST APIs, covering authentication, rate limiting, caching, database design, and deployment.", 500),
}


async def streaming_request(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Stream SSE and record per-token timestamps."""
    token_timestamps: list[float] = []
    t0 = time.perf_counter()

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        temperature=0.0,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token_timestamps.append(time.perf_counter())

    total_ms = (time.perf_counter() - t0) * 1000

    if not token_timestamps:
        return {"error": "no tokens received"}

    ttft_ms = (token_timestamps[0] - t0) * 1000
    inter_token_delays = [
        (token_timestamps[i] - token_timestamps[i - 1]) * 1000
        for i in range(1, len(token_timestamps))
    ]

    return {
        "mode": "streaming",
        "n_tokens": len(token_timestamps),
        "ttft_ms": round(ttft_ms, 1),
        "total_ms": round(total_ms, 1),
        "tpot_mean_ms": round(statistics.mean(inter_token_delays), 2) if inter_token_delays else 0,
        "tpot_p95_ms": round(sorted(inter_token_delays)[int(len(inter_token_delays) * 0.95)], 2) if inter_token_delays else 0,
        "tpot_max_ms": round(max(inter_token_delays), 2) if inter_token_delays else 0,
    }


async def non_streaming_request(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Non-streaming: wait for full response. No TTFT observable."""
    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=False,
        temperature=0.0,
    )
    total_ms = (time.perf_counter() - t0) * 1000
    n_tokens = response.usage.completion_tokens if response.usage else 0

    return {
        "mode": "non_streaming",
        "n_tokens": n_tokens,
        "total_ms": round(total_ms, 1),
        "ttft_ms": None,
        "tpot_mean_ms": round(total_ms / max(n_tokens, 1), 2),
    }


async def run_comparison(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    n_runs: int = 10,
) -> dict:
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")

    stream_results = []
    non_stream_results = []

    for _ in range(n_runs):
        s = await streaming_request(client, model, prompt, max_tokens)
        ns = await non_streaming_request(client, model, prompt, max_tokens)
        if "error" not in s:
            stream_results.append(s)
        non_stream_results.append(ns)

    await client.close()

    def agg(results: list[dict], key: str) -> float | None:
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(statistics.mean(vals), 1) if vals else None

    return {
        "streaming": {
            "ttft_mean_ms": agg(stream_results, "ttft_ms"),
            "total_mean_ms": agg(stream_results, "total_ms"),
            "tpot_mean_ms": agg(stream_results, "tpot_mean_ms"),
            "tpot_p95_ms": agg(stream_results, "tpot_p95_ms"),
        },
        "non_streaming": {
            "total_mean_ms": agg(non_stream_results, "total_ms"),
            "tpot_mean_ms": agg(non_stream_results, "tpot_mean_ms"),
        },
    }


async def measure_inter_token_pattern(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int = 200,
) -> dict:
    """
    Profile the inter-token delay pattern over a full generation.
    First token typically has higher latency (prefill), then decode is steady.
    """
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    token_timestamps: list[float] = []
    t0 = time.perf_counter()

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        temperature=0.0,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token_timestamps.append(time.perf_counter())

    await client.close()

    if len(token_timestamps) < 3:
        return {"error": "too few tokens"}

    delays = [(token_timestamps[i] - token_timestamps[i - 1]) * 1000 for i in range(1, len(token_timestamps))]
    first_10 = delays[:10]
    rest = delays[10:]

    return {
        "n_tokens": len(token_timestamps),
        "ttft_ms": round((token_timestamps[0] - t0) * 1000, 1),
        "early_tokens_mean_ms": round(statistics.mean(first_10), 2) if first_10 else 0,
        "steady_state_mean_ms": round(statistics.mean(rest), 2) if rest else 0,
        "steady_state_std_ms": round(statistics.stdev(rest), 2) if len(rest) > 1 else 0,
        "total_ms": round((token_timestamps[-1] - t0) * 1000, 1),
        "observation": (
            "TTFT > first inter-token delays (prefill dominates first token)"
            if (token_timestamps[0] - t0) * 1000 > (statistics.mean(first_10) if first_10 else 0)
            else "Fast prefill — decode latency dominates"
        ),
    }


async def main(args: argparse.Namespace):
    print("=== Phase 7 Lab 4: Streaming Inference ===\n")

    # Part 1: Streaming vs non-streaming across prompt sizes
    print("--- Part 1: Streaming vs Non-Streaming Latency ---")
    print(f"{'Prompt size':>12} {'TTFT (stream)':>16} {'Total (stream)':>16} {'Total (non-stream)':>20} {'Overhead':>10}")
    print("-" * 78)

    all_results = {}
    for size, (prompt, max_tokens) in PROMPTS.items():
        result = await run_comparison(args.url, args.model, prompt, max_tokens, n_runs=args.runs)
        all_results[size] = result

        stream_total = result["streaming"]["total_mean_ms"] or 0
        nostream_total = result["non_streaming"]["total_mean_ms"] or 0
        overhead = ((stream_total - nostream_total) / nostream_total * 100) if nostream_total else 0
        ttft = result["streaming"]["ttft_mean_ms"] or 0

        print(f"{size:>12} {ttft:>14.1f}ms {stream_total:>14.1f}ms {nostream_total:>18.1f}ms {overhead:>+8.1f}%")

    # Part 2: Inter-token delay pattern
    print("\n--- Part 2: Inter-Token Delay Pattern (medium prompt) ---")
    prompt, max_tokens = PROMPTS["medium"]
    pattern = await measure_inter_token_pattern(args.url, args.model, prompt, min(max_tokens, 150))

    if "error" not in pattern:
        print(f"  Tokens generated:          {pattern['n_tokens']}")
        print(f"  TTFT (first token):        {pattern['ttft_ms']}ms")
        print(f"  Early tokens mean (2-10):  {pattern['early_tokens_mean_ms']}ms/token")
        print(f"  Steady state mean (11+):   {pattern['steady_state_mean_ms']}ms/token  (±{pattern['steady_state_std_ms']}ms)")
        print(f"  Observation: {pattern['observation']}")

    print("\n--- Key Takeaways ---")
    medium = all_results.get("medium", {})
    ttft = medium.get("streaming", {}).get("ttft_mean_ms") or 0
    total = medium.get("streaming", {}).get("total_mean_ms") or 0
    if ttft and total:
        perceived_speedup = total / ttft
        print(f"  Streaming: user sees first token at {ttft:.0f}ms, perceives {perceived_speedup:.0f}× faster response")
    print("  Non-streaming: user waits for full completion — better for batch/offline use cases")
    print("  TPOT drives UI responsiveness; TTFT drives initial perceived latency")
    print("  Streaming overhead is typically minimal (<5%) due to SSE protocol efficiency")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"results": all_results, "pattern": pattern}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 Streaming Inference benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output")
    asyncio.run(main(parser.parse_args()))
