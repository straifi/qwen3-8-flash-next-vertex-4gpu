# `Qwen/Qwen3.8-Flash-Next` on a Reserved 4-GPU G4 Node (`g4-standard-192`) in `me-central2` — Deployment & Benchmark Report

> [!IMPORTANT]
> **Target Vertex AI Prediction Endpoint Configuration**
> - **Project**: `YOUR_GCP_PROJECT_ID`
> - **Region / Zone**: `me-central2` (Reservation in `me-central2-a`)
> - **Endpoint Display Name**: `qwen3-8-flash-next-g4-4gpu-ep`
> - **Model Registry Display Name**: `qwen3-8-flash-next-vllm-4gpu`
> - **Machine & GPU Configuration**: `g4-standard-192` with **4 × `NVIDIA_RTX_PRO_6000` Blackwell Server Edition** (`384 GB` total GDDR7 VRAM, `192` vCPUs, `720 GB` System RAM)
> - **Provisioning Mode**: **Reserved (`SPECIFIC_RESERVATION`)** pinned to `projects/YOUR_GCP_PROJECT_ID/zones/me-central2-a/reservations/YOUR_RESERVATION_NAME`
> - **vLLM Engine Configuration**: `TP=4, EP=True (--enable-expert-parallel), --engram-config={"cpu_offload":true}, --attention-config.indexer_kv_dtype=fp8, --max-model-len=32768`

---

## 1. Hardware & Memory Telemetry on 4 × NVIDIA RTX PRO 6000 (`g4-standard-192`)

### 1.1 Why `--engram-config={"cpu_offload":true}` is Essential on 4 GPUs
The full `Qwen/Qwen3.8-Flash-Next` checkpoint is **`335.30 GiB`** across 131 `.safetensors` shards:
- **CPU-Offloaded Engram N-Gram Embedding Table (`189.12 GiB`)**: Offloaded to the `g4-standard-192` host system RAM (`720 GB`), keeping GPU high-bandwidth GDDR7 memory dedicated to active neural layers and KV cache.
- **GPU-Resident Active Model Weights (`243.00 GiB` total = `60.75 GiB / GPU`)**: Routed MoE experts (`512` total, `10` active per token), shared experts, Gated DeltaNet linear attention layers, and full attention layers sharded across `GPUs 0, 1, 2, 3` via `TP=4` and `EP=True`.

### 1.2 Verified Per-GPU Memory Allocation (`TP=4, EP=True`)

| Physical Resource | Worker Role | Weights + Runtime | Peak Activation + CUDA Graph | FP8 KV Cache Reserved | Total Memory Used | Total Capacity |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **GPU 0 (`RTX PRO 6000`)** | `Worker_TP0_EP0` | `62.03 GiB` | `4.68 GiB` (`3.29 + 1.39`) | **`20.16 GiB`** | **`86.87 GiB`** | `94.97 GiB` |
| **GPU 1 (`RTX PRO 6000`)** | `Worker_TP1_EP1` | `62.03 GiB` | `4.68 GiB` (`3.29 + 1.39`) | **`20.16 GiB`** | **`86.87 GiB`** | `94.97 GiB` |
| **GPU 2 (`RTX PRO 6000`)** | `Worker_TP2_EP2` | `62.03 GiB` | `4.68 GiB` (`3.29 + 1.39`) | **`20.16 GiB`** | **`86.87 GiB`** | `94.97 GiB` |
| **GPU 3 (`RTX PRO 6000`)** | `Worker_TP3_EP3` | `62.03 GiB` | `4.68 GiB` (`3.29 + 1.39`) | **`20.16 GiB`** | **`86.87 GiB`** | `94.97 GiB` |
| **Host System RAM** | Engram CPU Offload | **`189.12 GiB`** | — | — | **`~205 GiB`** | `720.00 GiB` |

### 1.3 Startup & KV Cache Telemetry Logs
```text
(Worker_TP0..3) Offloading 189.12 GiB of Engram embedding weights to CPU memory
(Worker_TP0..3) Model loading took 60.75 GiB memory and 414.04 seconds
(Worker_TP0..3_EP0..3) Available KV cache memory: 20.16 GiB
(EngineCore) GPU KV cache size: 1,269,296 tokens
(EngineCore) Maximum concurrency for 32,768 tokens per request: 38.74x
(Worker_TP0..3_EP0..3) Graph capturing finished in 14 secs, took 1.39 GiB
```

---

## 2. End-to-End Inference & Throughput Benchmarks (`4 × RTX PRO 6000`, `TP=4`)

### 2.1 Single-Stream Functional & Generation Benchmarks

| # | Test Suite | Endpoint Method | Status | Output Tokens | Latency (s) | Throughput (tok/s) |
| :- | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | **Identity & Architectural Sanity Check** | `:predict` | **PASS** | `68` | `1.412s` | **`48.16 tok/s`** |
| 2 | **4-GPU VRAM Math & Reasoning** | `:predict` | **PASS** | `274` | `3.420s` | **`80.12 tok/s`** |
| 3 | **Python Coding Generation (Kahn's / Top-K Experts)** | `:predict` | **PASS** | `384` | `4.613s` | **`83.25 tok/s`** |
| 4 | **Multilingual Classical Arabic Poem + English Translation** | `:rawPredict` | **PASS** | `295` | `3.720s` | **`79.37 tok/s`** |

### 2.2 Multi-Concurrency Stress Test Benchmarks (`max_tokens=128` per request)

| Concurrency Level | Total Requests | Success / Fail | Wall Time (s) | Throughput (RPS) | Total Output Tokens | Aggregate Token Throughput (tok/s) | p50 Latency (s) | p90 Latency (s) | p99 Latency (s) | Mean Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4 | **4 / 0 (100%)** | `9.98s` | `0.40` | 512 | **51.33 tok/s** | `2.49s` | `2.53s` | `2.53s` | `2.49s` |
| **8** | 16 | **16 / 0 (100%)** | `18.31s` | `0.87` | 2,048 | **111.86 tok/s** | `3.00s` | `18.26s` | `18.30s` | `7.57s` |
| **16** | 32 | **32 / 0 (100%)** | `43.24s` | `0.74` | 4,096 | **94.73 tok/s** | `7.74s` | `38.26s` | `43.23s` | `14.64s` |
| **32** | 64 | **64 / 0 (100%)** | `49.09s` | `1.30` | 8,192 | **166.86 tok/s** | `3.43s` | `38.26s` | `49.07s` | `15.63s` |

---

## 3. Reproducing the Deployment & Benchmark in Your Environment

```bash
./deploy.sh \
  --project-id YOUR_GCP_PROJECT_ID \
  --bucket-name YOUR_GCS_BUCKET_NAME \
  --reservation-name YOUR_RESERVATION_NAME
```
