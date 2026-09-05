from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

@dataclass
class ScenarioDef:
    name: str
    system: str
    user: str
    expected_output_tokens: int = 256
    use_thinking: bool = False

    @classmethod
    def short(cls) -> "ScenarioDef":
        return cls(
            name="short",
            system=_SYSTEM_PROMPT,
            user="What is the capital of France? Give a one-sentence answer.",
            expected_output_tokens=30,
        )

    @classmethod
    def medium(cls) -> "ScenarioDef":
        return cls(
            name="medium",
            system=_SYSTEM_PROMPT,
            user=(
                "Compare REST and GraphQL APIs. Cover: data fetching, "
                "over/under-fetching, versioning, and when to choose each. "
                "Structure your response with clear headings."
            ),
            expected_output_tokens=300,
        )

    @classmethod
    def long(cls) -> "ScenarioDef":
        return cls(
            name="long",
            system=_SYSTEM_PROMPT,
            user=(
                "Explain how the transformer self-attention mechanism works. "
                "Cover: query/key/value projections, scaled dot-product attention, "
                "multi-head attention, positional encodings, and why attention "
                "replaced RNNs for sequence modelling. Include the mathematical "
                "formulation and intuitive explanations."
            ),
            expected_output_tokens=700,
        )

    @classmethod
    def complex(cls) -> "ScenarioDef":
        return cls(
            name="complex",
            system=_SYSTEM_PROMPT,
            user=(
                "Design a distributed rate-limiting system for an API gateway "
                "serving 10M requests/day across 20 global regions. Requirements: "
                "P99 latency < 5ms, eventual consistency acceptable within 100ms, "
                "support sliding-window and token-bucket algorithms, handle "
                "network partitions gracefully. Provide architecture diagram "
                "description, technology choices with justification, failure modes "
                "and mitigations, and a capacity estimate."
            ),
            expected_output_tokens=800,
            use_thinking=True,
        )

    @classmethod
    def all(cls) -> list["ScenarioDef"]:
        return [cls.short(), cls.medium(), cls.long(), cls.complex()]


_SYSTEM_PROMPT = (
    "You are an expert software engineer and systems architect. "
    "Your answers are technically precise, well-structured, and concise. "
    "You cite trade-offs, provide concrete examples, and avoid filler. "
    "When asked to compare technologies, use a balanced, evidence-based approach. "
    "When asked to design systems, consider scalability, reliability, and cost. "
    "Always satisfy every constraint stated in the question. "
    "Format longer responses with markdown headings and bullet points for clarity. "
    "Do not add unnecessary caveats or disclaimers unless they are technically relevant. "
    "If a question has a single best answer, give it directly before elaborating."
)


# ---------------------------------------------------------------------------
# Per-request result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    # Identity
    scenario: str
    backend: str        # "anthropic" | "openai" | "gemini" | "groq" | "together" | "vllm" | "sglang" | "tensorrt"
    model: str
    run_index: int

    # --- LATENCY ---
    ttft_ms: float                          # Time To First Token ≈ prefill proxy
    total_ms: float                         # End-to-end wall time
    tpot_ms: float                          # Time Per Output Token ≈ decode proxy
    prefill_ms: float = 0.0                 # Explicit prefill phase (open models via /metrics)
    decode_ms: float = 0.0                  # Explicit decode phase  (open models via /metrics)
    queue_time_ms: float = 0.0             # Time in server request queue

    # --- THROUGHPUT ---
    throughput_tps: float = 0.0            # Output tokens/sec for this request
    prefill_throughput_tps: float = 0.0    # Prompt tokens/sec during prefill
    decode_throughput_tps: float = 0.0     # Output tokens/sec during decode

    # --- BOTTLENECK ---
    bottleneck: str = ""                   # "prefill" | "decode" | "balanced" | "unknown"

    # --- TOKENS ---
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0             # closed model cache hits
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0

    # --- COST ---
    cost_usd: float = 0.0

    # --- GPU (open models + profiling) ---
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    gpu_power_watts: float = 0.0
    kv_cache_hit_rate: float = 0.0
    kv_cache_utilization: float = 0.0

    # --- CONTENT ---
    response_text: str = ""
    error: str | None = None
    request_id: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def __post_init__(self) -> None:
        if not self.bottleneck and (self.ttft_ms > 0 or self.prefill_ms > 0):
            self.bottleneck = classify_bottleneck(
                self.prefill_ms, self.decode_ms,
                self.ttft_ms, self.tpot_ms, self.output_tokens,
            )


# ---------------------------------------------------------------------------
# Bottleneck classification
# ---------------------------------------------------------------------------

def classify_bottleneck(
    prefill_ms: float,
    decode_ms: float,
    ttft_ms: float,
    tpot_ms: float,
    output_tokens: int,
) -> str:
    """
    Classify whether a request was prefill-bound or decode-bound.

    For open models: uses explicit prefill_ms / decode_ms from server metrics.
    For closed APIs: uses TTFT as prefill proxy and TPOT × tokens as decode proxy.

    Returns: "prefill" | "decode" | "balanced" | "unknown"
    """
    if prefill_ms > 0 and decode_ms > 0:
        total = prefill_ms + decode_ms
        ratio = prefill_ms / total
        if ratio > 0.6:
            return "prefill"
        if ratio < 0.4:
            return "decode"
        return "balanced"

    # Closed API proxy
    if ttft_ms > 0 and tpot_ms > 0 and output_tokens > 0:
        decode_total = tpot_ms * max(output_tokens - 1, 1)
        total = ttft_ms + decode_total
        ratio = ttft_ms / total
        if ratio > 0.6:
            return "prefill"
        if ratio < 0.4:
            return "decode"
        return "balanced"

    return "unknown"


# ---------------------------------------------------------------------------
# Aggregated statistics
# ---------------------------------------------------------------------------

@dataclass
class AggregatedMetrics:
    scenario: str
    backend: str
    model: str
    n: int                              # number of successful runs

    # Latency percentiles (ms)
    ttft_p50_ms: float
    ttft_p95_ms: float
    prefill_p50_ms: float               # 0.0 when not available (closed APIs)
    prefill_p95_ms: float
    decode_p50_ms: float
    decode_p95_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    total_p50_ms: float
    total_p95_ms: float
    queue_p50_ms: float

    # Throughput
    throughput_mean_tps: float
    throughput_p5_tps: float            # worst-case (p5)
    prefill_throughput_mean_tps: float
    decode_throughput_mean_tps: float

    # Tokens & cost
    input_tokens_mean: float
    output_tokens_mean: float
    cost_mean_usd: float
    cost_total_usd: float

    # Bottleneck summary
    dominant_bottleneck: str            # "prefill" | "decode" | "balanced" | "unknown"
    prefill_pct: float                  # fraction of runs classified prefill-bound
    decode_pct: float


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def aggregate(results: list[RunResult]) -> AggregatedMetrics:
    good = [r for r in results if r.succeeded]
    if not good:
        raise ValueError("No successful runs to aggregate")

    ttfts = [r.ttft_ms for r in good]
    tpots = [r.tpot_ms for r in good]
    totals = [r.total_ms for r in good]
    prefills = [r.prefill_ms for r in good]
    decodes = [r.decode_ms for r in good]
    queues = [r.queue_time_ms for r in good]
    tps = [r.throughput_tps for r in good]
    prefill_tps = [r.prefill_throughput_tps for r in good]
    decode_tps = [r.decode_throughput_tps for r in good]

    bottlenecks = [r.bottleneck for r in good]
    prefill_count = bottlenecks.count("prefill")
    decode_count = bottlenecks.count("decode")
    dominant = max(
        ["prefill", "decode", "balanced", "unknown"],
        key=lambda b: bottlenecks.count(b),
    )

    return AggregatedMetrics(
        scenario=good[0].scenario,
        backend=good[0].backend,
        model=good[0].model,
        n=len(good),
        ttft_p50_ms=_pct(ttfts, 50),
        ttft_p95_ms=_pct(ttfts, 95),
        prefill_p50_ms=_pct(prefills, 50),
        prefill_p95_ms=_pct(prefills, 95),
        decode_p50_ms=_pct(decodes, 50),
        decode_p95_ms=_pct(decodes, 95),
        tpot_p50_ms=_pct(tpots, 50),
        tpot_p95_ms=_pct(tpots, 95),
        total_p50_ms=_pct(totals, 50),
        total_p95_ms=_pct(totals, 95),
        queue_p50_ms=_pct(queues, 50),
        throughput_mean_tps=statistics.mean(tps),
        throughput_p5_tps=_pct(tps, 5),
        prefill_throughput_mean_tps=statistics.mean(prefill_tps),
        decode_throughput_mean_tps=statistics.mean(decode_tps),
        input_tokens_mean=statistics.mean(r.input_tokens for r in good),
        output_tokens_mean=statistics.mean(r.output_tokens for r in good),
        cost_mean_usd=statistics.mean(r.cost_usd for r in good),
        cost_total_usd=sum(r.cost_usd for r in good),
        dominant_bottleneck=dominant,
        prefill_pct=prefill_count / len(good),
        decode_pct=decode_count / len(good),
    )


# ---------------------------------------------------------------------------
# GPU snapshot (used by profiling layer)
# ---------------------------------------------------------------------------

@dataclass
class GPUSnapshot:
    gpu_index: int
    memory_used_mb: float
    memory_total_mb: float
    utilization_pct: float
    power_watts: float
    sm_clock_mhz: float
    memory_clock_mhz: float
    temperature_c: float
    timestamp_ms: float


@dataclass
class ServerMetrics:
    """Real-time metrics scraped from open model server /metrics endpoints."""
    kv_cache_hit_rate: float = 0.0
    kv_cache_utilization: float = 0.0
    num_running_requests: int = 0
    num_queued_requests: int = 0
    prefill_throughput_tps: float = 0.0
    decode_throughput_tps: float = 0.0
    avg_batch_size: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
