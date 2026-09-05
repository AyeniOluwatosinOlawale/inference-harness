from .base import BaseBackend
from .anthropic_backend import AnthropicBackend
from .openai_backend import OpenAIBackend
from .gemini_backend import GeminiBackend
from .groq_backend import GroqBackend
from .together_backend import TogetherBackend
from .vllm_backend import VLLMBackend
from .sglang_backend import SGLangBackend
from .trtllm_backend import TRTLLMBackend

__all__ = [
    "BaseBackend", "AnthropicBackend", "OpenAIBackend", "GeminiBackend",
    "GroqBackend", "TogetherBackend", "VLLMBackend", "SGLangBackend", "TRTLLMBackend",
]


def get_backend(
    backend: str,
    model: str,
    *,
    api_key: str | None = None,
    base_url: str = "",
    gpu_cost_per_hour: float = 0.0,
) -> BaseBackend:
    b = backend.lower()
    if b == "anthropic":
        return AnthropicBackend(model=model, api_key=api_key)
    if b == "openai":
        return OpenAIBackend(model=model, api_key=api_key)
    if b == "gemini":
        return GeminiBackend(model=model, api_key=api_key)
    if b == "groq":
        return GroqBackend(model=model, api_key=api_key)
    if b == "together":
        return TogetherBackend(model=model, api_key=api_key)
    if b == "vllm":
        return VLLMBackend(model=model, base_url=base_url or "http://localhost:8000",
                           gpu_cost_per_hour=gpu_cost_per_hour)
    if b == "sglang":
        return SGLangBackend(model=model, base_url=base_url or "http://localhost:30000",
                             gpu_cost_per_hour=gpu_cost_per_hour)
    if b in ("tensorrt", "trtllm", "trt"):
        return TRTLLMBackend(model=model, base_url=base_url or "http://localhost:8001",
                             gpu_cost_per_hour=gpu_cost_per_hour)
    raise ValueError(
        f"Unknown backend: {backend!r}. "
        "Choose from: anthropic, openai, gemini, groq, together, vllm, sglang, tensorrt"
    )
