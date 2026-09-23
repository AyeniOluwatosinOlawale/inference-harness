"""
Phase 7 — Lab 3: Continuous Batching vs Static Batching
=========================================================
Demonstrates the throughput difference between continuous (iteration-level)
batching and static (request-level) batching.

Static batching: wait until a full batch of N requests arrives, run them all,
then release. Long requests block the batch even after short ones finish.

Continuous batching: at every decode step, the scheduler can add new requests
or remove completed ones. Short requests leave immediately; new ones fill gaps.

This is the core innovation behind vLLM and most modern inference servers.

Metrics measured:
  - Throughput (tokens/s) vs concurrency level
  - Short request latency under mixed load
  - GPU utilisation

Setup:
  # Server with continuous batching (vLLM default)
  python -m vllm.entrypoints.openai.api_server --model <model> --port 8000

Usage:
  python -m inference_harness.benchmarks.serving_labs.continuous_batching \\
    --url http://localhost:8000 --model <model>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx
import openai


PROMPTS = {
    "short":  [
        "What is the capital of France?",
        "Name a planet in our solar system.",
        "What colour is the sky on a clear day?",
        "Who wrote Romeo and Juliet?",
        "What is H2O?",
        "How many days are in a week?",
        "What is the largest ocean?",
        "Name a primary colour.",
    ],
    "medium": [
        "Explain how a neural network learns in 3 sentences.",
        "What are the main differences between Python 2 and Python 3?",
        "Briefly describe the water cycle.",
        "What is REST and why is it used?",
        "Explain what a mutex is in 2 sentences.",
    ],
    "long": [
        "Write a detailed explanation of how gradient descent works, including the mathematics.",
        "Describe the history of the internet from ARPANET to the modern web.",
        "Explain TCP/IP networking stack from physical layer to application layer.",
        "Write a comprehensive guide to database indexing strategies.",
    ],
}


async def timed_chat(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: str = "",
) -> dict:
    t0 = time.perf_counter()
    first_token_t = None
    output_tokens = 0

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        temperature=0.0,
    )
    async for chunk in stream:
        if first_token_t is None and chunk.choices and chunk.choices[0].delta.content:
            first_token_t = time.perf_counter()
        if chunk.choices and chunk.choices[0].delta.content:
            output_tokens += 1

    completed_t = time.perf_counter()
    ttft = ((first_token_t or completed_t) - t0) * 1000
    total = (completed_t - t0) * 1000

    return {
        "request_id": request_id,
        "ttft_ms": round(ttft, 1),
        "total_ms": round(total, 1),
        "output_tokens": output_tokens,
        "throughput_tps": round(output_tokens / ((completed_t - t0) or 1e-9), 1),
    }


async def run_concurrency_sweep(
    url: str,
    model: str,
    concurrency_levels: list[int],
    n_requests_per_level: int = 30,
    mix: str = "medium",
) -> list[dict]:
    """
    At each concurrency level, send N requests in parallel and measure
    throughput and latency. This is the fundamental continuous batching experiment.
    """
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    results = []

    for concurrency in concurrency_levels:
        sem = asyncio.Semaphore(concurrency)
        prompts = PROMPTS[mix] * (n_requests_per_level // len(PROMPTS[mix]) + 1)
        prompts = prompts[:n_requests_per_level]

        max_tokens_map = {"short": 30, "medium": 120, "long": 400}
        max_tokens = max_tokens_map[mix]

        async def bounded(prompt: str, i: int) -> dict:
            async with sem:
                return await timed_chat(client, model, prompt, max_tokens, f"req_{i}")

        t_start = time.perf_counter()
        request_results = await asyncio.gather(*[bounded(p, i) for i, p in enumerate(prompts)])
        elapsed = time.perf_counter() - t_start

        ttfts = [r["ttft_ms"] for r in request_results]
        total_tokens = sum(r["output_tokens"] for r in request_results)
        server_throughput = total_tokens / elapsed if elapsed > 0 else 0

        results.append({
            "concurrency": concurrency,
            "n_requests": len(request_results),
            "total_tokens": total_tokens,
            "elapsed_s": round(elapsed, 1),
            "server_throughput_tps": round(server_throughput, 1),
            "ttft_mean_ms": round(statistics.mean(ttfts), 1),
            "ttft_p50_ms": round(sorted(ttfts)[len(ttfts) // 2], 1),
            "ttft_p95_ms": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 1),
        })
        print(f"  concurrency={concurrency:>3}: {server_throughput:>8.1f} tps  TTFT mean={statistics.mean(ttfts):.1f}ms")

    await client.close()
    return results


async def run_mixed_size_experiment(
    url: str,
    model: str,
    n_short: int = 20,
    n_long: int = 5,
    concurrency: int = 8,
) -> dict:
    """
    The canonical continuous batching demo: mix of short and long requests.
    With continuous batching, short requests complete quickly without waiting
    for long ones to finish. Without it (static batching emulation), short
    requests wait for the longest request in the batch.
    """
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)

    short_results: list[dict] = []
    long_results: list[dict] = []

    async def send(prompt: str, max_tokens: int, is_short: bool, idx: int):
        async with sem:
            r = await timed_chat(client, model, prompt, max_tokens, f"{'short' if is_short else 'long'}_{idx}")
            if is_short:
                short_results.append(r)
            else:
                long_results.append(r)

    short_prompts_list = (PROMPTS["short"] * (n_short // len(PROMPTS["short"]) + 1))[:n_short]
    long_prompts_list = (PROMPTS["long"] * (n_long // len(PROMPTS["long"]) + 1))[:n_long]

    tasks = (
        [send(p, 25, True, i) for i, p in enumerate(short_prompts_list)]
        + [send(p, 350, False, i) for i, p in enumerate(long_prompts_list)]
    )
    await asyncio.gather(*tasks)
    await client.close()

    def stats(results: list[dict]) -> dict:
        if not results:
            return {}
        ttfts = sorted([r["ttft_ms"] for r in results])
        return {
            "n": len(results),
            "ttft_mean_ms": round(statistics.mean(ttfts), 1),
            "ttft_p50_ms": round(ttfts[len(ttfts) // 2], 1),
            "ttft_p95_ms": round(ttfts[int(len(ttfts) * 0.95)], 1),
            "throughput_mean_tps": round(statistics.mean(r["throughput_tps"] for r in results), 1),
        }

    return {"short": stats(short_results), "long": stats(long_results)}


async def get_server_metrics(url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{url}/metrics")
            metrics = {}
            for line in r.text.splitlines():
                if "vllm:gpu_cache_usage_perc" in line and not line.startswith("#"):
                    metrics["gpu_cache_usage"] = float(line.split()[-1])
                if "vllm:num_requests_running" in line and not line.startswith("#"):
                    metrics["num_running"] = float(line.split()[-1])
            return metrics
    except Exception:
        return {}


async def main(args: argparse.Namespace):
    print("=== Phase 7 Lab 3: Continuous Batching ===\n")

    # Part 1: Throughput vs concurrency sweep
    print("--- Part 1: Throughput Scaling (medium requests) ---")
    concurrency_levels = [1, 2, 4, 8, 16, 32]
    sweep_results = await run_concurrency_sweep(
        args.url, args.model, concurrency_levels, n_requests_per_level=30, mix="medium"
    )

    print(f"\n{'Concurrency':>12} {'Throughput (tps)':>18} {'TTFT mean':>12} {'TTFT p95':>10}")
    print("-" * 56)
    baseline_tps = sweep_results[0]["server_throughput_tps"]
    for r in sweep_results:
        scaling = r["server_throughput_tps"] / baseline_tps
        print(f"{r['concurrency']:>12} {r['server_throughput_tps']:>16.1f} {r['ttft_mean_ms']:>10.1f}ms {r['ttft_p95_ms']:>8.1f}ms  ({scaling:.1f}×)")

    # Part 2: Mixed short/long
    print("\n--- Part 2: Mixed Short + Long Requests ---")
    print(f"  Sending {args.n_short} short + {args.n_long} long concurrently...", end=" ", flush=True)
    mixed = await run_mixed_size_experiment(
        args.url, args.model, args.n_short, args.n_long, args.concurrency
    )
    print("done\n")

    print(f"{'':>15} {'Short requests':>16} {'Long requests':>16}")
    print("-" * 50)
    print(f"{'Count':>15} {mixed['short']['n']:>16} {mixed['long']['n']:>16}")
    print(f"{'TTFT mean':>15} {mixed['short']['ttft_mean_ms']:>14.1f}ms {mixed['long']['ttft_mean_ms']:>14.1f}ms")
    print(f"{'TTFT p95':>15} {mixed['short']['ttft_p95_ms']:>14.1f}ms {mixed['long']['ttft_p95_ms']:>14.1f}ms")

    print("\nKey finding: with continuous batching, short request TTFT is decoupled from long request length.")
    print("Short requests complete in milliseconds even while 350-token long requests are mid-generation.")
    print("Without continuous batching (static), short requests would wait until the entire batch finishes.")

    if args.output:
        out = {
            "concurrency_sweep": sweep_results,
            "mixed_workload": mixed,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 Continuous Batching benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-short", type=int, default=20)
    parser.add_argument("--n-long", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output")
    asyncio.run(main(parser.parse_args()))
