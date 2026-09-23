# Phase 7 — Model Serving Labs

Three labs covering the core mechanics of production LLM serving.
All labs work with any OpenAI-compatible server (vLLM, SGLang, TRT-LLM).

## Setup

```bash
# Start vLLM (works for all labs)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 512

# For chunked prefill lab: also start a baseline server (no chunked prefill)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --disable-chunked-prefill
```

---

## Lab 2 — Chunked Prefill

**File:** `chunked_prefill.py`

Measures the TTFT vs throughput tradeoff of `--enable-chunked-prefill`.

```bash
python -m inference_harness.benchmarks.serving_labs.chunked_prefill \
  --baseline-url http://localhost:8001 \
  --chunked-url  http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-long 4 --n-short 20 --concurrency 8
```

**What to look for:**
- Short request TTFT drops significantly with chunked prefill (long prefills no longer monopolise the GPU)
- Long request TTFT increases slightly (chunked = more iterations)
- Overall throughput stays similar

---

## Lab 3 — Continuous Batching

**File:** `continuous_batching.py`

Shows throughput scaling vs concurrency and mixed short/long request behaviour.

```bash
python -m inference_harness.benchmarks.serving_labs.continuous_batching \
  --url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-short 20 --n-long 5 --concurrency 8
```

**What to look for:**
- Throughput scales roughly linearly up to the saturation point
- Short request TTFT remains low even alongside long requests
- This is the fundamental behaviour that makes vLLM/SGLang efficient

---

## Lab 4 — Streaming Inference

**File:** `streaming.py`

Profiles streaming SSE vs non-streaming, and measures inter-token delay patterns.

```bash
python -m inference_harness.benchmarks.serving_labs.streaming \
  --url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --runs 10
```

**What to look for:**
- TTFT in streaming mode is far lower than total latency — this is what users perceive
- Inter-token delay is stable after the first few tokens (prefill phase settles)
- Streaming overhead is minimal (<5%)

---

## Run All Labs

```bash
MODEL="Qwen/Qwen2.5-7B-Instruct"

# Lab 2
python -m inference_harness.benchmarks.serving_labs.chunked_prefill \
  --baseline-url http://localhost:8001 --chunked-url http://localhost:8000 \
  --model $MODEL --output results_chunked_prefill.json

# Lab 3
python -m inference_harness.benchmarks.serving_labs.continuous_batching \
  --url http://localhost:8000 --model $MODEL --output results_continuous_batching.json

# Lab 4
python -m inference_harness.benchmarks.serving_labs.streaming \
  --url http://localhost:8000 --model $MODEL --output results_streaming.json
```
