# inference-harness

A unified inference engineering harness for benchmarking, profiling, and comparing LLM inference across cloud APIs and self-hosted open models.

```
Inference Harness
      │
┌─────┴──────────┐
▼                ▼
Closed APIs      Open Models
Anthropic        vLLM · SGLang · TensorRT-LLM
OpenAI           (OpenAI-compatible HTTP)
Gemini
Groq
Together AI
      │                │
      ▼                ▼
 API Benchmark    GPU Benchmark
      │                │
      └────────┬────────┘
               ▼
        Compare Results
    Latency · Throughput
    Prefill · Decode · Bottleneck
```

---

## Features

- **8 backends** — Anthropic Claude, OpenAI GPT, Google Gemini, Groq, Together AI, vLLM, SGLang, TensorRT-LLM
- **Latency profiling** — TTFT, TPOT, total latency, queue time (p50 / p95 percentiles)
- **Throughput profiling** — output tok/s, prefill tok/s, decode tok/s
- **Bottleneck detection** — classifies each run as prefill-bound, decode-bound, or balanced with actionable recommendations
- **GPU profiling** — pynvml (SM clock, HBM clock, power, temperature), nvidia-smi fallback, Prometheus /metrics scraping (KV cache hit rate, server queue depth)
- **CUDA profiling** — Nsight Systems subprocess + NVTX range annotations (opt-in)
- **Concurrency sweep** — measure TTFT degradation and throughput scaling under load
- **Cross-backend comparison** — side-by-side Δ% tables for TTFT, throughput, and cost
- **Live web dashboard** — FastAPI + WebSocket with real-time Chart.js plots
- **Export** — JSON and CSV output for every run

---

## Installation

```bash
git clone https://github.com/your-org/inference-harness
cd inference-harness
pip install -e .

# Optional: CUDA profiling support
pip install -e ".[cuda]"
# Also install Nsight Systems CLI: https://developer.nvidia.com/nsight-systems
```

### Requirements

- Python ≥ 3.10
- API keys for whichever closed model backends you use
- A running vLLM / SGLang / TensorRT-LLM server for open model benchmarks

---

## Quick Start

### Benchmark a closed model API

```bash
inference-harness run \
  --backend anthropic \
  --model claude-sonnet-4-6 \
  --scenario medium \
  --runs 5
```

### Benchmark an open model (vLLM)

```bash
inference-harness run \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000 \
  --runs 5
```

### Compare closed vs open model

```bash
inference-harness compare \
  --closed anthropic \
  --closed-model claude-sonnet-4-6 \
  --open vllm \
  --open-model meta-llama/Llama-3.1-8B-Instruct \
  --open-url http://localhost:8000 \
  --runs 5
```

### Launch the web dashboard

```bash
inference-harness dashboard --port 8080
# Open http://localhost:8080
```

---

## CLI Reference

### `run` — Single backend benchmark

```
inference-harness run [OPTIONS]

  --backend        TEXT     Backend: anthropic | openai | gemini | groq | together | vllm | sglang | tensorrt
  --model          TEXT     Model name / ID
  --scenario       TEXT     short | medium | long | complex  (default: all four)
  --runs           INT      Runs per scenario  (default: 3)
  --base-url       TEXT     Base URL for open model servers  (default: http://localhost:8000)
  --api-key        TEXT     API key (overrides env var)
  --gpu-cost-per-hour FLOAT GPU instance cost USD/hr for amortized cost tracking  (default: 0.0)
  --concurrency           Run a concurrency sweep after baseline
  --conc-levels    TEXT    Comma-separated levels  (default: 1,2,4,8,16)
  --profile        TEXT    none | system | cuda | both  (default: system)
  --no-gpu-metrics        Skip GPU profiling
  --output         PATH    Save results to JSON
  --csv            PATH    Save aggregated metrics to CSV
```

### `compare` — Side-by-side comparison

```
inference-harness compare [OPTIONS]

  --closed         TEXT    Closed model backend name
  --closed-model   TEXT    Closed model ID
  --open           TEXT    Open model backend: vllm | sglang | tensorrt
  --open-model     TEXT    Open model name
  --open-url       TEXT    Open model server URL  (default: http://localhost:8000)
  --scenario       TEXT    Scenario filter (default: all)
  --runs           INT     Runs per scenario  (default: 3)
  --output         PATH    Save results to JSON
```

### `dashboard` — Web dashboard

```
inference-harness dashboard [OPTIONS]

  --port  INT   Port number  (default: 8080)
  --host  TEXT  Bind address  (default: 0.0.0.0)
```

---

## Environment Variables

```bash
# Closed model API keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=gsk_...
TOGETHER_API_KEY=...

# Open model server
OPEN_MODEL_URL=http://localhost:8000
OPEN_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

# GPU cost tracking
GPU_COST_PER_HOUR=2.50
```

---

## Python Library Usage

```python
import asyncio
from inference_harness import Harness, ScenarioDef

async def main():
    # Benchmark Anthropic
    async with Harness("anthropic", "claude-sonnet-4-6") as h:
        results = await h.run(ScenarioDef.medium(), runs=5)
        agg = h.aggregate(results)
        print(f"TTFT p50: {agg.ttft_p50_ms:.1f}ms")
        print(f"Throughput: {agg.throughput_mean_tps:.1f} tok/s")
        print(f"Bottleneck: {agg.dominant_bottleneck}")

    # Benchmark vLLM
    async with Harness("vllm", "meta-llama/Llama-3.1-8B",
                        base_url="http://localhost:8000",
                        profiling="system") as open_h:
        open_results = await open_h.run(ScenarioDef.medium(), runs=5)
        open_agg = open_h.aggregate(open_results)

    # Compare
    closed_results, open_results = await h.compare(open_h, ScenarioDef.medium())

asyncio.run(main())
```

---

## Metrics Reference

### Latency

| Metric | Description | Source |
|---|---|---|
| `ttft_ms` | Time To First Token | Measured (first stream chunk) |
| `total_ms` | End-to-end wall time | Measured |
| `tpot_ms` | Time Per Output Token | `(total - ttft) / output_tokens` |
| `prefill_ms` | Explicit prefill phase duration | Open models: from `/metrics` |
| `decode_ms` | Explicit decode phase duration | Open models: from `/metrics` |
| `queue_time_ms` | Time waiting in server queue | vLLM: `request_queue_time` |

### Throughput

| Metric | Description |
|---|---|
| `throughput_tps` | Output tokens / sec (single request) |
| `prefill_throughput_tps` | Prompt tokens processed / sec during prefill |
| `decode_throughput_tps` | Output tokens generated / sec during decode |

### Bottleneck Classification

```
if prefill_ms and decode_ms available (open models):
    ratio = prefill_ms / (prefill_ms + decode_ms)
else (closed APIs — use proxies):
    ratio = ttft_ms / (ttft_ms + tpot_ms × output_tokens)

ratio > 0.6  →  PREFILL-bound
ratio < 0.4  →  DECODE-bound
otherwise    →  BALANCED
```

**Prefill-bound** — long prompt processing dominates. Caused by large context windows, no prefix caching, or insufficient compute parallelism.

**Decode-bound** — token generation dominates. Caused by memory-bandwidth saturation during KV cache reads (the common case for large models).

### Recommendations

| Bottleneck | Optimizations |
|---|---|
| DECODE | Speculative decoding · INT4/FP8 quantization · Larger batch size · Prefix caching · PagedAttention |
| PREFILL | Chunked prefill · Increase tensor parallelism · Prefix caching for shared system prompts · FlashAttention-3 |

### GPU Metrics (open models)

| Metric | Source |
|---|---|
| `gpu_memory_used_mb` | pynvml / nvidia-smi |
| `gpu_utilization_pct` | pynvml / nvidia-smi |
| `gpu_power_watts` | pynvml |
| `kv_cache_hit_rate` | vLLM: `/metrics` · SGLang: `/get_server_info` · TRT-LLM: `/metrics` |
| `kv_cache_utilization` | Prometheus endpoint |

---

## Supported Backends

### Closed Model APIs

| Backend | Flag | Default Model | Auth |
|---|---|---|---|
| Anthropic | `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| Google Gemini | `gemini` | `gemini-2.0-flash` | `GOOGLE_API_KEY` |
| Groq | `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Together AI | `together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` |

### Open Model Backends

| Backend | Flag | Default URL | Metrics Source |
|---|---|---|---|
| vLLM | `vllm` | `http://localhost:8000` | `GET /metrics` (Prometheus) |
| SGLang | `sglang` | `http://localhost:30000` | `GET /get_server_info` (JSON) |
| TensorRT-LLM | `tensorrt` | `http://localhost:8001` | `GET /metrics` (Prometheus) |

---

## Launching Open Model Servers

Docker Compose templates are included in `docker/`.

### vLLM

```bash
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=hf_... \
docker compose -f docker/vllm.yaml up -d

# Wait for healthy, then benchmark
inference-harness run --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000
```

### SGLang

```bash
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
docker compose -f docker/sglang.yaml up -d

inference-harness run --backend sglang \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:30000
```

### TensorRT-LLM

```bash
TRTLLM_MODEL_DIR=./models \
docker compose -f docker/trtllm.yaml up -d

inference-harness run --backend tensorrt \
  --model llama-3.1-8b \
  --base-url http://localhost:8001
```

---

## Web Dashboard

Start the dashboard, then run any benchmark in a separate terminal. Results stream live via WebSocket.

```bash
# Terminal 1
inference-harness dashboard --port 8080

# Terminal 2
inference-harness run --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --runs 20
```

Open `http://localhost:8080` to see:
- Live TTFT line chart (per request)
- Throughput bar chart
- Prefill vs Decode stacked phase chart
- GPU utilization area chart
- Bottleneck badge per request (PREFILL / DECODE / BALANCED)

---

## Output Format

### Terminal

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Baseline: Latency, Throughput & Bottleneck                                  │
├─────────┬──────────┬────┬──────────┬──────────┬────────────┬────────────┬───┤
│ Scenario│ Backend  │ N  │ TTFT p50 │ TTFT p95 │ Prefil p50 │ Decode p50 │...│
├─────────┼──────────┼────┼──────────┼──────────┼────────────┼────────────┼───┤
│ short   │ anthropic│  5 │   312ms  │   389ms  │      —     │      —     │...│
│ medium  │ vllm     │  5 │   180ms  │   220ms  │   145ms    │   820ms    │...│
└─────────┴──────────┴────┴──────────┴──────────┴────────────┴────────────┴───┘

  BOTTLENECK ANALYSIS
  short    Prefill: 312ms (TTFT proxy)  Decode: TPOT×tok proxy  → ✓ BALANCED
  medium   Prefill: 145ms              Decode: 820ms             → ⚠ DECODE

  DECODE-bound (medium):
    → Enable speculative decoding (draft model)
    → Increase batch size to amortize HBM reads
    → Quantize weights (INT4/FP8) to reduce memory bandwidth
    → Enable PagedAttention / prefix caching
```

### JSON export (`--output results.json`)

```json
{
  "metadata": {
    "timestamp": "2025-09-05T12:00:00Z",
    "backend": "vllm",
    "model": "meta-llama/Llama-3.1-8B-Instruct"
  },
  "aggregated": [
    {
      "scenario": "medium",
      "backend": "vllm",
      "ttft_p50_ms": 180.2,
      "prefill_p50_ms": 145.1,
      "decode_p50_ms": 820.4,
      "throughput_mean_tps": 28.5,
      "prefill_throughput_mean_tps": 1240.0,
      "decode_throughput_mean_tps": 28.5,
      "dominant_bottleneck": "decode",
      "decode_pct": 1.0
    }
  ],
  "runs": [ ... ]
}
```

---

## Project Structure

```
inference-harness/
├── pyproject.toml
├── .env.example
├── docker/
│   ├── vllm.yaml
│   ├── sglang.yaml
│   └── trtllm.yaml
└── src/inference_harness/
    ├── __init__.py               # Public API
    ├── __main__.py               # CLI (typer)
    ├── types.py                  # RunResult, AggregatedMetrics, ScenarioDef
    ├── harness.py                # Harness — main library class
    ├── report.py                 # Terminal tables, JSON/CSV export
    ├── backends/
    │   ├── base.py               # BaseBackend (ABC)
    │   ├── anthropic_backend.py
    │   ├── openai_backend.py
    │   ├── gemini_backend.py
    │   ├── groq_backend.py
    │   ├── together_backend.py
    │   ├── vllm_backend.py
    │   ├── sglang_backend.py
    │   └── trtllm_backend.py
    ├── profiling/
    │   ├── system_profiler.py    # pynvml + nvidia-smi + Prometheus
    │   └── cuda_profiler.py      # Nsight Systems + NVTX
    ├── benchmarks/
    │   ├── runner.py             # run_scenario, concurrency sweep
    │   └── optimizations/        # Caching, batching, routing (extensible)
    └── api/
        ├── server.py             # FastAPI + WebSocket
        └── ws_manager.py         # WebSocket connection manager
```

---

## License

MIT
