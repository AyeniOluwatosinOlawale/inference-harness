"""
inference-harness — Unified inference engineering harness.

Supports: Anthropic, OpenAI, Gemini, Groq, Together AI, vLLM, SGLang, TensorRT-LLM.

Quick start:
    from inference_harness import Harness, ScenarioDef
    h = Harness("anthropic", "claude-sonnet-4-6")
    results = await h.run(ScenarioDef.medium(), runs=3)
"""
__version__ = "0.1.0"

from .harness import Harness
from .types import (
    AggregatedMetrics,
    GPUSnapshot,
    RunResult,
    ScenarioDef,
    ServerMetrics,
    aggregate,
    classify_bottleneck,
)
from .backends import get_backend

__all__ = [
    "Harness",
    "get_backend",
    "RunResult",
    "AggregatedMetrics",
    "ScenarioDef",
    "GPUSnapshot",
    "ServerMetrics",
    "aggregate",
    "classify_bottleneck",
]
