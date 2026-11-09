#!/usr/bin/env python3
"""
End-to-End Functional Verification and Multi-Concurrency Performance Benchmark Suite
for Qwen/Qwen3.8-Flash-Next deployed on a Reserved 4-GPU G4 Node (`g4-standard-192`,
4x NVIDIA RTX PRO 6000) on Vertex AI Prediction in `me-central2`.

Tests both `:predict` and `:rawPredict` endpoints and runs multi-concurrency load waves
(Concurrency = 1, 8, 16, 32).

Usage:
  python3 scripts/stress_test_endpoint.py <ENDPOINT_ID> [REGION] [OUT_PATH]
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import google.auth
import google.auth.transport.requests

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = REPO_ROOT / "deployment_config.json"


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


CFG = load_config()
GCP_CFG = CFG.get("gcp", {})
STAGE_CFG = CFG.get("pipeline_stages", {})

PROJECT_ID = os.environ.get("PROJECT_ID") or GCP_CFG.get("project_id", "")
if not PROJECT_ID or PROJECT_ID == "YOUR_GCP_PROJECT_ID":
    PROJECT_ID = subprocess.check_output(
        ["gcloud", "config", "get-value", "project"], text=True
    ).strip()

ENDPOINT_ID = sys.argv[1] if len(sys.argv) > 1 else (os.environ.get("ENDPOINT_ID") or STAGE_CFG.get("existing_endpoint_id", ""))
REGION = sys.argv[2] if len(sys.argv) > 2 else (os.environ.get("REGION") or (GCP_CFG.get("regions") or ["me-central2"])[0])
OUT_PATH = (
    sys.argv[3]
    if len(sys.argv) > 3
    else str(SCRIPT_DIR / "stress_test_results.json")
)

if not ENDPOINT_ID:
    state_file = SCRIPT_DIR / "active_deployment.json"
    if state_file.exists():
        st = json.loads(state_file.read_text(encoding="utf-8"))
        ENDPOINT_ID = st.get("endpoint_id", "")
        REGION = st.get("region", REGION)

if not ENDPOINT_ID:
    sys.exit("Usage: python3 scripts/stress_test_endpoint.py <ENDPOINT_ID> [REGION] [OUT_PATH]")

PREDICT_URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
RAW_PREDICT_URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:rawPredict"


def get_access_token() -> str:
    try:
        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        if creds.token:
            return creds.token
    except Exception:
        pass
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def send_predict_request(token: str, prompt: str, max_tokens: int = 256, temperature: float = 0.6):
    payload = {
        "instances": [
            {
                "messages": [
                    {"role": "system", "content": "You are Qwen3.8-Flash-Next, a helpful AI assistant. Be concise."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    t0 = time.perf_counter()
    last_err = ""
    for attempt in range(3):
        req = urllib.request.Request(
            PREDICT_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "vertex-ai-stress-tester/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8")
                latency = time.perf_counter() - t0
                body = json.loads(raw)
                pred = body.get("predictions", [{}])[0]
                if isinstance(pred, dict):
                    usage = pred.get("usage", {})
                    comp_tokens = usage.get("completion_tokens", 0)
                    if not comp_tokens:
                        content_str = str(pred.get("content", "")) + str(pred.get("reasoning_content", ""))
                        comp_tokens = max(1, len(content_str.split()))
                else:
                    comp_tokens = max(1, len(str(pred).split()))
                return {"ok": True, "latency": latency, "tokens": comp_tokens, "response": pred}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_err = f"HTTP {e.code}: {err_body[:300]}"
            if e.code in (429, 502, 503, 504) and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            latency = time.perf_counter() - t0
            return {"ok": False, "latency": latency, "tokens": 0, "error": last_err}
        except Exception as e:
            last_err = str(e)
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            latency = time.perf_counter() - t0
            return {"ok": False, "latency": latency, "tokens": 0, "error": last_err}


def send_raw_predict_request(token: str, prompt: str, max_tokens: int = 128, temperature: float = 0.2):
    payload = {
        "messages": [
            {"role": "system", "content": "You are Qwen3.8-Flash-Next, a helpful AI assistant. Be concise."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        RAW_PREDICT_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "vertex-ai-stress-tester/1.0",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
            latency = time.perf_counter() - t0
            body = json.loads(raw)
            usage = body.get("usage", {})
            comp_tokens = usage.get("completion_tokens", 0)
            return {"ok": True, "latency": latency, "tokens": comp_tokens, "response": body}
    except urllib.error.HTTPError as e:
        latency = time.perf_counter() - t0
        err_body = e.read().decode("utf-8", errors="replace")
        return {"ok": False, "latency": latency, "tokens": 0, "error": f"HTTP {e.code}: {err_body}"}
    except Exception as e:
        latency = time.perf_counter() - t0
        return {"ok": False, "latency": latency, "tokens": 0, "error": str(e)}


def run_e2e_checks(token: str):
    print("=" * 72)
    print(f"1. END-TO-END FUNCTIONAL VERIFICATION (Endpoint: {ENDPOINT_ID} in {REGION})")
    print("=" * 72)

    test_cases = [
        (
            "Identity & Sanity Check (:predict)",
            "In one sentence, state your model architecture capabilities and confirm you are running online.",
            512,
        ),
        (
            "4-GPU VRAM Math & Reasoning (:predict)",
            "Solve step-by-step: A g4-standard-192 node has 4 NVIDIA RTX PRO 6000 GPUs with 96 GB VRAM each. "
            "After offloading the 189.12 GB Engram table to host RAM, model weights use 60.75 GB per GPU. "
            "How many GB of VRAM remain per GPU and across all 4 GPUs?",
            768,
        ),
        (
            "Coding Generation (:predict)",
            "Write a fast Python function `top_k_experts(logits, k=10)` using PyTorch that returns the indices and softmax weights of the top-k routed experts.",
            1536,
        ),
    ]

    e2e_results = []
    for name, prompt, max_tok in test_cases:
        print(f"\n--- Test: {name} ---", flush=True)
        res = send_predict_request(token, prompt, max_tokens=max_tok, temperature=0.2)
        print(f"Status: {'PASS' if res['ok'] else 'FAIL'} | Latency: {res['latency']:.2f}s | Output Tokens: {res['tokens']}", flush=True)
        if res["ok"]:
            print(f"Response preview: {json.dumps(res['response'], ensure_ascii=False)[:1000]}", flush=True)
        else:
            print(f"Error: {res.get('error')}", flush=True)
        e2e_results.append({"test": name, **res})

    print("\n--- Test: OpenAI Chat Completions via :rawPredict ---", flush=True)
    raw_res = send_raw_predict_request(
        token,
        "Explain in two sentences why Mixture-of-Experts with PLE N-gram CPU offload is efficient on 4x RTX PRO 6000 GPUs.",
        max_tokens=1024,
        temperature=0.2,
    )
    print(f"Status: {'PASS' if raw_res['ok'] else 'FAIL'} | Latency: {raw_res['latency']:.2f}s | Output Tokens: {raw_res['tokens']}", flush=True)
    if raw_res["ok"]:
        print(f"Response preview: {json.dumps(raw_res['response'], ensure_ascii=False)[:1000]}", flush=True)
    else:
        print(f"Error: {raw_res.get('error')}", flush=True)
    e2e_results.append({"test": "OpenAI Chat Completions (:rawPredict)", **raw_res})
    return e2e_results


def run_stress_wave(token: str, concurrency: int, total_requests: int, max_tokens: int = 128):
    prompts = [
        "Explain how Mixture-of-Experts (MoE) routing reduces active parameter FLOPs during LLM decoding.",
        "Write a Python async worker pool using asyncio.Queue that limits concurrency to N tasks.",
        "Compute the derivative of f(x) = x^3 * ln(x) + exp(-2x) and simplify.",
        "Summarize the key differences between Tensor Parallelism and Expert Parallelism in 4-GPU serving.",
        "Explain why FP8 KV-cache quantization doubles effective context capacity on Blackwell GPUs.",
        "Write a SQL query to find the top 3 highest-traffic endpoints per region over the last 24 hours.",
        "Explain how speculative decoding with Multi-Token Prediction (MTP) speeds up autoregressive generation.",
        "Describe how N-gram embedding lookup tables can be offloaded to CPU RAM with asynchronous PCIe prefetching.",
    ]

    print(f"\n>>> Running Stress Wave: Concurrency={concurrency}, Total Requests={total_requests}, MaxTokens={max_tokens} <<<", flush=True)
    t_start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(send_predict_request, token, prompts[i % len(prompts)], max_tokens, 0.7)
            for i in range(total_requests)
        ]
        for f in futures:
            results.append(f.result())
    wall_time = time.perf_counter() - t_start

    ok_res = [r for r in results if r["ok"]]
    latencies = sorted([r["latency"] for r in ok_res])
    total_tokens = sum(r["tokens"] for r in ok_res)

    def pct(arr, p):
        if not arr:
            return 0.0
        idx = min(len(arr) - 1, int(len(arr) * p / 100.0))
        return arr[idx]

    summary = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful": len(ok_res),
        "failed": len(results) - len(ok_res),
        "wall_time_s": round(wall_time, 2),
        "rps": round(len(ok_res) / wall_time, 2) if wall_time > 0 else 0,
        "total_output_tokens": total_tokens,
        "output_tps": round(total_tokens / wall_time, 2) if wall_time > 0 else 0,
        "p50_latency_s": round(pct(latencies, 50), 2),
        "p90_latency_s": round(pct(latencies, 90), 2),
        "p99_latency_s": round(pct(latencies, 99), 2),
        "mean_latency_s": round(statistics.mean(latencies), 2) if latencies else 0.0,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    token = get_access_token()
    e2e = run_e2e_checks(token)
    print("\n" + "=" * 72, flush=True)
    print("2. MULTI-CONCURRENCY STRESS TEST SUITE", flush=True)
    print("=" * 72, flush=True)
    waves = [
        (1, 4, 128),
        (8, 16, 128),
        (16, 32, 128),
        (32, 64, 128),
    ]
    all_summaries = []
    for conc, reqs, mtok in waves:
        token = get_access_token()
        all_summaries.append(run_stress_wave(token, conc, reqs, mtok))
    with open(OUT_PATH, "w") as f:
        json.dump(
            {
                "region": REGION,
                "endpoint_id": ENDPOINT_ID,
                "e2e": e2e,
                "stress_waves": all_summaries,
            },
            f,
            indent=2,
        )
    print(f"\nSaved E2E and stress test results to {OUT_PATH}", flush=True)
