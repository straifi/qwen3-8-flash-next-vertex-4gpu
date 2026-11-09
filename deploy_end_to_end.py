#!/usr/bin/env python3
"""
Unified One-Command End-to-End Deployment Script for Qwen/Qwen3.8-Flash-Next (335.3 GiB MoE)
on a Reserved 4-GPU G4 Node (`g4-standard-192`, 4x NVIDIA RTX PRO 6000) in `me-central2` (`me-central2-a`).

Driven by `deployment_config.json` with optional CLI flag overrides:
  ./deploy.sh
  # or:
  python3 deploy_end_to_end.py --config deployment_config.json \
    --project-id YOUR_GCP_PROJECT_ID \
    --bucket-name YOUR_GCS_BUCKET_NAME \
    --reservation-name YOUR_RESERVATION_NAME

Implements all 5 stages end-to-end:
  Stage 1: GitHub Bundled Model Artifacts (`model_weights/`) + Parallel Shard Streaming -> Customer GCS (`me-central2`)
  Stage 2: Custom Vertex AI vLLM Serving Container Build -> Artifact Registry (`me-central2`)
  Stage 3: Vertex AI Model Upload, Endpoint Creation & Reserved 4-GPU Deployment (`me-central2` / `me-central2-a`)
  Stage 4: Live Endpoint End-to-End Functional Verification (`:predict` & `:rawPredict`)
  Stage 5: Multi-Concurrency Performance Benchmark (`Concurrency = 1, 8, 16, 32`)
"""

import argparse
import concurrent.futures
import datetime
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.auth
import google.auth.transport.requests


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "deployment_config.json"

G4_MACHINE_GPU_MAP = {
    "g4-standard-48": 1,
    "g4-standard-96": 2,
    "g4-standard-192": 4,
    "g4-standard-384": 8,
}


# ==============================================================================
# Authentication & REST Helpers
# ==============================================================================

def normalize_hf_model_id(hf_input: str) -> str:
    """Accepts either 'Qwen/Qwen3.8-Flash-Next' or 'https://huggingface.co/Qwen/Qwen3.8-Flash-Next'."""
    cleaned = hf_input.strip().rstrip("/")
    cleaned = re.sub(r"^https?://(www\.)?huggingface\.co/", "", cleaned)
    parts = cleaned.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    raise ValueError(
        f"Invalid Hugging Face model identifier '{hf_input}'. "
        "Expected 'org/model' or 'https://huggingface.co/org/model'."
    )


def get_gcp_token() -> str:
    try:
        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        if creds.token:
            return creds.token
    except Exception:
        pass
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def run_cmd(cmd: List[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def api_request(
    method: str,
    url: str,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    token = get_gcp_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "vertex-ai-qwen38-4gpu-deployer/1.0",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {method} {url}: {err_text}") from e


# ==============================================================================
# GCE Reservation Discovery & Validation (`me-central2-a`)
# ==============================================================================

def resolve_reservation_affinity(
    project_id: str,
    capacity_type: str,
    affinity_type: str,
    reservation_name: str,
    reservation_zone: str,
    reservation_project_id: str,
    machine_type: str,
) -> Optional[Dict[str, Any]]:
    """
    Constructs Vertex AI MachineSpec.reservationAffinity for a Compute Engine reservation
    (e.g. a reserved g4-standard-192 4-GPU node in me-central2-a).
    """
    cap = capacity_type.lower()
    aff = affinity_type.upper() if affinity_type else "SPECIFIC_RESERVATION"

    if cap == "spot" or aff == "NO_RESERVATION":
        return {"reservationAffinityType": "NO_RESERVATION"} if aff == "NO_RESERVATION" else None

    if aff == "ANY_RESERVATION":
        return {"reservationAffinityType": "ANY_RESERVATION"}

    if cap != "reserved" and aff != "SPECIFIC_RESERVATION":
        return None

    res_proj = reservation_project_id.strip() if reservation_project_id else project_id
    if not res_proj or res_proj == "YOUR_GCP_PROJECT_ID":
        res_proj = project_id
    res_zone = reservation_zone.strip() if reservation_zone else "me-central2-a"
    raw_name = (reservation_name or "").strip()

    # If user already provided a full resource name: projects/.../zones/.../reservations/...
    if raw_name.startswith("projects/"):
        full_res_name = raw_name
        print(f"[Reservation] Using full reservation resource path: {full_res_name}", flush=True)
        return {
            "reservationAffinityType": "SPECIFIC_RESERVATION",
            "key": "compute.googleapis.com/reservation-name",
            "values": [full_res_name],
        }

    # If placeholder or AUTO_DISCOVER, query gcloud compute reservations in reservation_zone
    if not raw_name or raw_name in ("YOUR_RESERVATION_NAME", "AUTO_DISCOVER", "AUTO"):
        print(
            f"[Reservation] Attempting to auto-discover READY reservations in zone '{res_zone}' (project '{res_proj}')...",
            flush=True,
        )
        disc = run_cmd(
            [
                "gcloud", "compute", "reservations", "list",
                f"--project={res_proj}",
                f"--filter=zone ~ {res_zone}",
                "--format=json",
            ],
            check=False,
        )
        candidates = []
        if disc.returncode == 0 and disc.stdout.strip():
            try:
                res_list = json.loads(disc.stdout)
                for r in res_list:
                    r_name = r.get("name", "")
                    r_status = r.get("status", "")
                    r_mt = r.get("specificReservation", {}).get("instanceProperties", {}).get("machineType", "")
                    r_count = r.get("specificReservation", {}).get("count", "0")
                    r_in_use = r.get("specificReservation", {}).get("inUseCount", "0")
                    print(
                        f"  -> Found reservation '{r_name}': status={r_status}, machineType={r_mt}, "
                        f"count={r_count}, inUse={r_in_use}",
                        flush=True,
                    )
                    if r_status == "READY" and (not r_mt or r_mt == machine_type or "g4-" in r_mt):
                        candidates.append(r_name)
            except Exception as exc:
                print(f"  [Reservation auto-discovery parse note]: {exc}", flush=True)

        if len(candidates) == 1:
            raw_name = candidates[0]
            print(f"[Reservation] Auto-selected reservation '{raw_name}' in {res_zone}!", flush=True)
        elif len(candidates) > 1:
            sys.exit(
                f"ERROR: Multiple READY G4 reservations found in {res_zone}: {candidates}.\n"
                f"Please specify which one to use via --reservation-name <NAME> or set compute.reservation.name in deployment_config.json."
            )
        else:
            sys.exit(
                f"ERROR: Reservation name is still set to placeholder '{raw_name or 'YOUR_RESERVATION_NAME'}' "
                f"and no READY reservation was auto-discovered in {res_zone} (project={res_proj}).\n"
                "Please set compute.reservation.name in deployment_config.json or pass --reservation-name <YOUR_RESERVATION_NAME>."
            )

    full_res_name = f"projects/{res_proj}/zones/{res_zone}/reservations/{raw_name}"
    print(f"[Reservation] Targeting specific reservation: {full_res_name}", flush=True)
    return {
        "reservationAffinityType": "SPECIFIC_RESERVATION",
        "key": "compute.googleapis.com/reservation-name",
        "values": [full_res_name],
    }


# ==============================================================================
# Stage 1: GitHub Bundled Weights + Parallel Shard Streaming -> Customer GCS
# ==============================================================================

def stage1_transfer_weights_to_gcs(
    project_id: str,
    region: str,
    hf_model_id: str,
    bucket_name: str,
    gcs_prefix: str,
    num_workers: int,
    local_weights_dir: Path,
    source_gcs_uri: str = "",
    transfer_mode: str = "auto",
    hf_token: str = "",
) -> str:
    print("\n" + "=" * 80)
    print(f"STAGE 1: Staging Model Weights -> Customer GCS (gs://{bucket_name}/{gcs_prefix} in {region})")
    print("=" * 80)

    check_bucket = run_cmd(
        ["gcloud", "storage", "buckets", "describe", f"gs://{bucket_name}", f"--project={project_id}"],
        check=False,
    )
    if check_bucket.returncode != 0:
        print(f"Creating GCS bucket gs://{bucket_name} in {region}...", flush=True)
        run_cmd(
            [
                "gcloud", "storage", "buckets", "create", f"gs://{bucket_name}",
                f"--project={project_id}",
                f"--location={region}",
                "--uniform-bucket-level-access",
            ],
            capture=False,
        )
    else:
        print(f"GCS bucket gs://{bucket_name} already exists.", flush=True)

    gs_dest = f"gs://{bucket_name}/{gcs_prefix}"

    # If a source GCS URI is provided, copy directly in the cloud
    if source_gcs_uri and source_gcs_uri.startswith("gs://"):
        src = source_gcs_uri.rstrip("/") + "/*"
        print(f"Copying model artifacts directly from {src} to {gs_dest}/ ...", flush=True)
        run_cmd(["gcloud", "storage", "cp", "-r", src, f"{gs_dest}/"], capture=False)
        return gs_dest

    # Step 1A: Upload bundled config, tokenizer, vocab, and shard-index files from ./model_weights/ in this GitHub repo
    manifest_path = local_weights_dir / "shard_manifest.json"
    total_shards = 131
    expected_shard_sizes: Dict[str, int] = {}

    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards_meta = manifest_data.get("safetensors_shards", [])
        if shards_meta:
            total_shards = len(shards_meta)
            expected_shard_sizes = {s["filename"]: int(s["size_bytes"]) for s in shards_meta}
        print(
            f"Loaded bundled manifest {manifest_path}: {manifest_data.get('bundled_config_files_count', 12)} "
            f"config/tokenizer files + {total_shards} .safetensors shards ({manifest_data.get('total_size_gib', 335.3)} GiB).",
            flush=True,
        )

    if local_weights_dir.is_dir():
        bundled_files = [
            str(p)
            for p in sorted(local_weights_dir.iterdir())
            if p.is_file() and p.name not in ("shard_manifest.json", "stage_weights_to_gcs.py") and not p.name.endswith(".py")
        ]
        if bundled_files:
            print(
                f"Uploading {len(bundled_files)} pre-packaged config/tokenizer/index files from "
                f"{local_weights_dir} to {gs_dest}/ ...",
                flush=True,
            )
            run_cmd(["gcloud", "storage", "cp"] + bundled_files + [f"{gs_dest}/"], capture=False)

    # Step 1B: Check how many .safetensors shards are already present in gs://<bucket_name>/<gcs_prefix>/
    existing_ls = run_cmd(
        ["gcloud", "storage", "ls", "-l", f"{gs_dest}/model-*.safetensors"],
        check=False,
    )
    existing_shards = set()
    if existing_ls.returncode == 0:
        for line in existing_ls.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0].isdigit():
                fsize = int(parts[0])
                fname = parts[-1].rstrip("/").split("/")[-1]
                min_expected = expected_shard_sizes.get(fname, 1_000_000)
                if fsize >= min_expected:
                    existing_shards.add(fname)

    if len(existing_shards) >= total_shards:
        print(
            f"All {total_shards} .safetensors weight shards are already staged in {gs_dest}/! Skipping download.",
            flush=True,
        )
        return gs_dest

    print(
        f"Found {len(existing_shards)}/{total_shards} shards in {gs_dest}/. "
        f"Streaming remaining {total_shards - len(existing_shards)} shards...",
        flush=True,
    )

    transfer_script = f"""\
import os
import time
from huggingface_hub import HfApi, hf_hub_download
from google.cloud import storage

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

REPO_ID = os.environ["HF_MODEL_ID"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
GCS_PREFIX = os.environ.get("GCS_PREFIX", "vertex_model_ready").strip("/")
START_SHARD = int(os.environ.get("START_SHARD", "1"))
END_SHARD = int(os.environ.get("END_SHARD", "{total_shards}"))
TOTAL_SHARDS = int(os.environ.get("TOTAL_SHARDS", "{total_shards}"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None

LOCAL_DIR = "/workspace/hf_tmp" if os.path.isdir("/workspace") else "/tmp/hf_tmp"
os.makedirs(LOCAL_DIR, exist_ok=True)


def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    if START_SHARD == 1:
        api = HfApi(token=HF_TOKEN)
        repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
        config_files = [
            f for f in repo_files
            if not f.endswith(".safetensors") and not f.startswith(".")
        ]
        for cfg_file in config_files:
            blob_name = f"{{GCS_PREFIX}}/{{cfg_file}}"
            blob = bucket.blob(blob_name)
            if blob.exists():
                continue
            local_cfg = hf_hub_download(
                repo_id=REPO_ID,
                filename=cfg_file,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
            )
            blob.upload_from_filename(local_cfg, timeout=600)
            if os.path.exists(local_cfg):
                os.remove(local_cfg)

    print(f"Starting shard transfer for {{REPO_ID}} shards {{START_SHARD}}..{{END_SHARD}} -> gs://{{BUCKET_NAME}}/{{GCS_PREFIX}}/")
    for i in range(START_SHARD, END_SHARD + 1):
        filename = f"model-{{i:05d}}-of-{{TOTAL_SHARDS:05d}}.safetensors"
        blob_name = f"{{GCS_PREFIX}}/{{filename}}"
        blob = bucket.blob(blob_name)

        if blob.exists():
            blob.reload()
            if blob.size and blob.size > 1_000_000:
                print(f"[SKIP] {{filename}} already exists in GCS ({{blob.size / 1e9:.2f}} GB)")
                continue

        t0 = time.time()
        print(f"[DOWNLOADING] ({{i}}/{{TOTAL_SHARDS}}) {{filename}}...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )
        dl_time = time.time() - t0
        fsize = os.path.getsize(local_path)
        print(f"[UPLOADING] {{filename}} ({{fsize / 1e9:.2f}} GB, downloaded in {{dl_time:.1f}}s) -> gs://{{BUCKET_NAME}}/{{blob_name}}...")

        t1 = time.time()
        blob.chunk_size = 64 * 1024 * 1024
        blob.upload_from_filename(local_path, timeout=1800)
        ul_time = time.time() - t1
        print(f"[DONE] {{filename}} uploaded in {{ul_time:.1f}}s")

        if os.path.exists(local_path):
            os.remove(local_path)

    print("All requested shards transferred successfully!")


if __name__ == "__main__":
    main()
"""

    if transfer_mode.lower() == "local":
        print("Running local batch shard streamer...", flush=True)
        local_job_script = SCRIPT_DIR / "hf_transfer_job" / "transfer_shards.py"
        env = os.environ.copy()
        env["REPO_ID"] = hf_model_id
        env["PROJECT_ID"] = project_id
        env["BUCKET_NAME"] = bucket_name
        env["GCS_PREFIX"] = gcs_prefix
        if hf_token:
            env["HF_TOKEN"] = hf_token
        subprocess.run(
            [sys.executable, "-u", str(local_job_script), "1", str(total_shards), str(total_shards)],
            check=True,
            env=env,
        )
        return gs_dest

    cloudbuild_yaml = """\
steps:
- name: 'python:3.11-slim'
  entrypoint: 'bash'
  args:
  - '-c'
  - |
    pip install --no-cache-dir huggingface_hub hf_transfer google-cloud-storage &&
    HF_MODEL_ID=${_HF_MODEL_ID} \\
    BUCKET_NAME=${_BUCKET_NAME} \\
    GCS_PREFIX=${_GCS_PREFIX} \\
    START_SHARD=${_START_SHARD} \\
    END_SHARD=${_END_SHARD} \\
    TOTAL_SHARDS=${_TOTAL_SHARDS} \\
    python3 -u transfer_shards.py
options:
  machineType: 'E2_HIGHCPU_32'
  diskSizeGb: 200
timeout: '14400s'
substitutions:
  _HF_MODEL_ID: 'Qwen/Qwen3.8-Flash-Next'
  _BUCKET_NAME: 'qwen3-8-flash-next-weights'
  _GCS_PREFIX: 'vertex_model_ready'
  _START_SHARD: '1'
  _END_SHARD: '131'
  _TOTAL_SHARDS: '131'
"""

    with tempfile.TemporaryDirectory(prefix="hf_transfer_") as tmpdir:
        Path(tmpdir, "transfer_shards.py").write_text(transfer_script, encoding="utf-8")
        Path(tmpdir, "cloudbuild.yaml").write_text(cloudbuild_yaml, encoding="utf-8")

        workers = max(1, min(num_workers, max(1, total_shards)))
        chunk_size = max(1, math.ceil(total_shards / workers))
        build_ids: List[str] = []

        for w in range(workers):
            start_shard = w * chunk_size + 1
            end_shard = min(total_shards, (w + 1) * chunk_size)
            if start_shard > total_shards:
                break
            subs = (
                f"_HF_MODEL_ID={hf_model_id},"
                f"_BUCKET_NAME={bucket_name},"
                f"_GCS_PREFIX={gcs_prefix},"
                f"_START_SHARD={start_shard},"
                f"_END_SHARD={end_shard},"
                f"_TOTAL_SHARDS={total_shards}"
            )
            print(f"Submitting Cloud Build worker {w + 1}/{workers} (shards {start_shard}..{end_shard})...", flush=True)
            res = run_cmd(
                [
                    "gcloud", "builds", "submit", tmpdir,
                    f"--project={project_id}",
                    "--config", str(Path(tmpdir, "cloudbuild.yaml")),
                    f"--substitutions={subs}",
                    "--async",
                    "--format=value(id)",
                ],
                check=False,
            )
            if res.returncode != 0:
                print(
                    f"[WARN] Cloud Build submit returned non-zero ({res.stderr.strip()}). "
                    "Falling back to local batch shard transfer...",
                    flush=True,
                )
                local_job_script = SCRIPT_DIR / "hf_transfer_job" / "transfer_shards.py"
                env = os.environ.copy()
                env["REPO_ID"] = hf_model_id
                env["PROJECT_ID"] = project_id
                env["BUCKET_NAME"] = bucket_name
                env["GCS_PREFIX"] = gcs_prefix
                if hf_token:
                    env["HF_TOKEN"] = hf_token
                subprocess.run(
                    [sys.executable, "-u", str(local_job_script), "1", str(total_shards), str(total_shards)],
                    check=True,
                    env=env,
                )
                return gs_dest

            bid = res.stdout.strip().splitlines()[-1].strip()
            build_ids.append(bid)
            print(f"  -> Cloud Build ID: {bid}", flush=True)

        print(f"Polling {len(build_ids)} Cloud Build transfer worker(s)...", flush=True)
        while True:
            statuses = {}
            for bid in build_ids:
                r = run_cmd(
                    ["gcloud", "builds", "describe", bid, f"--project={project_id}", "--format=value(status)"]
                )
                statuses[bid] = r.stdout.strip()
            print("  [Shard Transfer Status] " + ", ".join(f"{b[:8]}={s}" for b, s in statuses.items()), flush=True)
            failed = [b for b, s in statuses.items() if s in ("FAILURE", "INTERNAL_ERROR", "TIMEOUT", "CANCELLED")]
            if failed:
                raise RuntimeError(f"HF->GCS Cloud Build worker(s) failed: {failed}")
            if all(s == "SUCCESS" for s in statuses.values()):
                print("All 143 model files (12 configs + 131 shards, 335.3 GiB) are staged in GCS!", flush=True)
                break
            time.sleep(20)

    return gs_dest


# ==============================================================================
# Stage 2: Build & Push Custom Vertex AI vLLM Container (`me-central2`)
# ==============================================================================

def stage2_build_vllm_container(
    project_id: str,
    region: str,
    ar_repo: str,
    image_uri: str,
    local_docker_build: bool = False,
) -> str:
    print("\n" + "=" * 80)
    print(f"STAGE 2: Building Custom Vertex AI vLLM Container -> {image_uri}")
    print("=" * 80)

    # Derive Artifact Registry location from image_uri if formatted as <region>-docker.pkg.dev/...
    ar_location = region
    if "-docker.pkg.dev/" in image_uri:
        ar_location = image_uri.split("-docker.pkg.dev/")[0].split("/")[-1]

    check_ar = run_cmd(
        [
            "gcloud", "artifacts", "repositories", "describe", ar_repo,
            f"--project={project_id}",
            f"--location={ar_location}",
        ],
        check=False,
    )
    if check_ar.returncode != 0:
        print(f"Creating Artifact Registry Docker repository '{ar_repo}' in {ar_location}...", flush=True)
        run_cmd(
            [
                "gcloud", "artifacts", "repositories", "create", ar_repo,
                f"--project={project_id}",
                "--repository-format=docker",
                f"--location={ar_location}",
            ],
            capture=False,
        )

    container_dir = SCRIPT_DIR / "vllm_container"
    if not container_dir.is_dir():
        raise FileNotFoundError(f"Expected container directory at {container_dir}")

    if local_docker_build:
        print(f"Building and pushing container locally via Docker -> {image_uri}...", flush=True)
        run_cmd(["gcloud", "auth", "configure-docker", f"{ar_location}-docker.pkg.dev", "--quiet"], capture=False)
        run_cmd(["docker", "build", "-t", image_uri, str(container_dir)], capture=False)
        run_cmd(["docker", "push", image_uri], capture=False)
        return image_uri

    print(f"Submitting Cloud Build for {container_dir} -> {image_uri}...", flush=True)
    cb_res = run_cmd(
        [
            "gcloud", "builds", "submit", str(container_dir),
            f"--project={project_id}",
            f"--tag={image_uri}",
            "--machine-type=e2-highcpu-32",
            "--timeout=3600s",
        ],
        check=False,
        capture=False,
    )
    if cb_res.returncode != 0:
        print("[WARN] Cloud Build failed or is restricted; falling back to local Docker build & push...", flush=True)
        run_cmd(["gcloud", "auth", "configure-docker", f"{ar_location}-docker.pkg.dev", "--quiet"], capture=False)
        run_cmd(["docker", "build", "-t", image_uri, str(container_dir)], capture=False)
        run_cmd(["docker", "push", image_uri], capture=False)

    print(f"Successfully built and pushed container: {image_uri}", flush=True)
    return image_uri


# ==============================================================================
# Stage 3: Vertex AI Model Upload, Endpoint Creation & Reserved 4-GPU Deployment
# ==============================================================================

def ensure_model_in_region(
    project_id: str,
    region: str,
    model_display_name: str,
    served_model_name: str,
    gcs_model_uri: str,
    image_uri: str,
    tp_size: int,
    enable_expert_parallel: bool,
    engram_cpu_offload: bool,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_cache_dtype: str,
    reasoning_parser: str,
    tool_call_parser: str,
    enable_auto_tool_choice: bool,
    trust_remote_code: bool,
    shared_memory_size_mb: int,
    deployment_timeout_seconds: int,
) -> str:
    list_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/models"
    res = api_request("GET", list_url)
    for m in res.get("models", []):
        if m.get("displayName") == model_display_name:
            print(f"[{region}] Reusing existing Vertex AI Model: {m['name']}", flush=True)
            return m["name"]

    vllm_args = [
        f"--model={gcs_model_uri}",
        f"--served-model-name={served_model_name}",
        "--host=0.0.0.0",
        "--port=8080",
        f"--tensor-parallel-size={tp_size}",
        f"--attention-config.indexer_kv_dtype={kv_cache_dtype}",
        f"--gpu-memory-utilization={gpu_memory_utilization}",
        f"--max-model-len={max_model_len}",
    ]
    if reasoning_parser:
        vllm_args.append(f"--reasoning-parser={reasoning_parser}")
    if enable_auto_tool_choice:
        vllm_args.append("--enable-auto-tool-choice")
    if tool_call_parser:
        vllm_args.append(f"--tool-call-parser={tool_call_parser}")
    if trust_remote_code:
        vllm_args.append("--trust-remote-code")
    if enable_expert_parallel:
        vllm_args.append("--enable-expert-parallel")
    if engram_cpu_offload:
        vllm_args.append('--engram-config={"cpu_offload":true}')

    print(f"[{region}] Uploading model '{model_display_name}' to Vertex AI Model Registry...", flush=True)
    upload_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/models:upload"
    model_body = {
        "model": {
            "displayName": model_display_name,
            "containerSpec": {
                "imageUri": image_uri,
                "args": vllm_args,
                "ports": [{"containerPort": 8080}],
                "predictRoute": "/predict",
                "healthRoute": "/health",
                "sharedMemorySizeMb": str(shared_memory_size_mb),
                "deploymentTimeout": f"{deployment_timeout_seconds}s",
            },
        }
    }
    u_op = api_request("POST", upload_url, body=model_body)
    u_op_name = u_op["name"]
    while True:
        time.sleep(3)
        op_state = api_request("GET", f"https://{region}-aiplatform.googleapis.com/v1/{u_op_name}")
        if op_state.get("done"):
            if "error" in op_state:
                raise RuntimeError(f"[{region}] Model upload error: {op_state['error']}")
            model_res_name = op_state["response"]["model"]
            print(f"[{region}] Uploaded model: {model_res_name}", flush=True)
            return model_res_name


def ensure_endpoint_in_region(
    project_id: str,
    region: str,
    endpoint_display_name: str,
) -> str:
    list_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints"
    res = api_request("GET", list_url)
    for ep in res.get("endpoints", []):
        if ep.get("displayName") == endpoint_display_name:
            ep_id = ep["name"].split("/")[-1]
            print(f"[{region}] Reusing existing Vertex AI Endpoint: {ep['name']} (id={ep_id})", flush=True)
            return ep_id

    print(f"[{region}] Creating Vertex AI Endpoint '{endpoint_display_name}'...", flush=True)
    create_op = api_request("POST", list_url, body={"displayName": endpoint_display_name})
    c_op_name = create_op["name"]
    while True:
        time.sleep(3)
        op_state = api_request("GET", f"https://{region}-aiplatform.googleapis.com/v1/{c_op_name}")
        if op_state.get("done"):
            if "error" in op_state:
                raise RuntimeError(f"[{region}] Endpoint creation error: {op_state['error']}")
            ep_name = op_state["response"]["name"]
            ep_id = ep_name.split("/")[-1]
            print(f"[{region}] Created Endpoint: {ep_name} (id={ep_id})", flush=True)
            return ep_id


def deploy_in_region(
    project_id: str,
    region: str,
    model_display_name: str,
    endpoint_display_name: str,
    served_model_name: str,
    gcs_model_uri: str,
    image_uri: str,
    machine_type: str,
    accelerator_type: str,
    gpu_count: int,
    capacity_type: str,
    reservation_affinity: Optional[Dict[str, Any]],
    min_replicas: int,
    max_replicas: int,
    shared_memory_size_mb: int,
    deployment_timeout_seconds: int,
    tp_size: int,
    enable_expert_parallel: bool,
    engram_cpu_offload: bool,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_cache_dtype: str,
    reasoning_parser: str,
    tool_call_parser: str,
    enable_auto_tool_choice: bool,
    trust_remote_code: bool,
) -> Tuple[bool, str, str, str]:
    model_res_name = ensure_model_in_region(
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        served_model_name=served_model_name,
        gcs_model_uri=gcs_model_uri,
        image_uri=image_uri,
        tp_size=tp_size,
        enable_expert_parallel=enable_expert_parallel,
        engram_cpu_offload=engram_cpu_offload,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
        reasoning_parser=reasoning_parser,
        tool_call_parser=tool_call_parser,
        enable_auto_tool_choice=enable_auto_tool_choice,
        trust_remote_code=trust_remote_code,
        shared_memory_size_mb=shared_memory_size_mb,
        deployment_timeout_seconds=deployment_timeout_seconds,
    )
    ep_id = ensure_endpoint_in_region(
        project_id=project_id,
        region=region,
        endpoint_display_name=endpoint_display_name,
    )

    ep_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints/{ep_id}"
    deploy_url = f"{ep_url}:deployModel"

    machine_spec: Dict[str, Any] = {
        "machineType": machine_type,
        "acceleratorType": accelerator_type,
        "acceleratorCount": gpu_count,
    }
    if reservation_affinity:
        machine_spec["reservationAffinity"] = reservation_affinity

    dedicated_resources: Dict[str, Any] = {
        "machineSpec": machine_spec,
        "minReplicaCount": min_replicas,
        "maxReplicaCount": max_replicas,
    }
    if capacity_type.lower() == "spot":
        dedicated_resources["spot"] = True

    deploy_body = {
        "deployedModel": {
            "model": model_res_name,
            "displayName": f"{model_display_name}-{machine_type}",
            "dedicatedResources": dedicated_resources,
            "disableContainerLogging": False,
        },
        "trafficSplit": {"0": 100},
    }

    res_desc = (
        reservation_affinity.get("values", [reservation_affinity.get("reservationAffinityType")])[0]
        if reservation_affinity
        else capacity_type.upper()
    )
    print(
        f"=== [{region}] Deploying {model_res_name} to Endpoint {ep_id} "
        f"({machine_type}, {gpu_count}x {accelerator_type}, capacity={capacity_type.upper()}, reservation={res_desc}) ===",
        flush=True,
    )
    d_op = api_request("POST", deploy_url, body=deploy_body)
    op_name = d_op["name"]
    print(f"[{region}] Deploy LRO started: {op_name}", flush=True)

    state_file = SCRIPT_DIR / "scripts" / "active_deployment.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps(
            {
                "project_id": project_id,
                "region": region,
                "endpoint_id": ep_id,
                "model_name": model_res_name,
                "operation_name": op_name,
                "machine_type": machine_type,
                "accelerator_count": gpu_count,
                "capacity_type": capacity_type,
                "reservation_affinity": reservation_affinity,
                "status": "DEPLOYING",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    seen_ts = set()
    start_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    while True:
        time.sleep(25)
        try:
            op = api_request("GET", f"https://{region}-aiplatform.googleapis.com/v1/{op_name}")
            done = op.get("done", False)
            stage = op.get("metadata", {}).get("deploymentStage", "UNKNOWN")
            now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%SZ")
            print(f"[{now_str}][{region}] done={done}, stage={stage}", flush=True)

            log_body = {
                "resourceNames": [f"projects/{project_id}"],
                "filter": (
                    f'resource.type="aiplatform.googleapis.com/Endpoint" '
                    f'AND resource.labels.endpoint_id="{ep_id}" '
                    f'AND timestamp>="{start_iso}"'
                ),
                "orderBy": "timestamp desc",
                "pageSize": 25,
            }
            logs = api_request("POST", "https://logging.googleapis.com/v2/entries:list", body=log_body)
            for e in reversed(logs.get("entries", [])):
                ts = e.get("timestamp", "")
                msg = e.get("textPayload") or json.dumps(e.get("jsonPayload", {}))
                key = (ts, msg[:120])
                if key not in seen_ts:
                    seen_ts.add(key)
                    print(f"  [{ts}] {msg.strip()[:280]}", flush=True)

            if done:
                if "error" in op:
                    print(f"[{region}] DEPLOYMENT ERROR: {json.dumps(op['error'], indent=2)}", flush=True)
                    return False, model_res_name, ep_id, ""

                deployed_model_id = op.get("response", {}).get("deployedModel", {}).get("id", "")
                print(
                    f"[{region}] DEPLOYMENT SUCCEEDED! endpoint_id={ep_id}, deployedModelId={deployed_model_id}",
                    flush=True,
                )
                state_file.write_text(
                    json.dumps(
                        {
                            "project_id": project_id,
                            "region": region,
                            "endpoint_id": ep_id,
                            "model_name": model_res_name,
                            "deployed_model_id": deployed_model_id,
                            "operation_name": op_name,
                            "machine_type": machine_type,
                            "accelerator_count": gpu_count,
                            "capacity_type": capacity_type,
                            "reservation_affinity": reservation_affinity,
                            "status": "DEPLOYED",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return True, model_res_name, ep_id, deployed_model_id
        except Exception as e:
            print(f"  [{region} monitor warning] {e}", flush=True)


# ==============================================================================
# Stage 4: Live Endpoint Verification (:predict & :rawPredict)
# ==============================================================================

def stage4_verify_endpoint(
    project_id: str,
    region: str,
    endpoint_id: str,
    served_model_name: str,
    custom_prompt: str,
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print(f"STAGE 4: Verifying Live Vertex AI Endpoint ({endpoint_id} in {region})")
    print("=" * 80)

    base_ep_url = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{region}/endpoints/{endpoint_id}"
    )
    results: List[Dict[str, Any]] = []

    predict_tests = [
        (
            "Test 1: Identity & Sanity Check (:predict)",
            "Reply in one short sentence confirming your model name and that you are running online on Vertex AI.",
            256,
        ),
        (
            "Test 2: 4-GPU RTX PRO 6000 VRAM Math & Reasoning (:predict)",
            "Solve step-by-step: A g4-standard-192 node has 4 NVIDIA RTX PRO 6000 GPUs with 96 GB VRAM each. "
            "After offloading the 189.12 GB Engram table to host RAM, model weights use 60.75 GB per GPU. "
            "How much total VRAM across all 4 GPUs remains available for KV cache and activations?",
            512,
        ),
        (
            "Test 3: Python Coding Generation (:predict)",
            "Write a clean Python function `top_k_experts(logits, k=10)` using PyTorch with type annotations "
            "that returns the indices and normalized softmax weights of the top-k routed experts.",
            1024,
        ),
    ]

    for title, prompt_text, max_tok in predict_tests:
        print(f"\n[{title}] Sending native Vertex AI `:predict` request...", flush=True)
        predict_payload = {
            "instances": [
                {
                    "model": served_model_name,
                    "messages": [
                        {"role": "system", "content": "You are Qwen3.8-Flash-Next running on Vertex AI. Be concise and accurate."},
                        {"role": "user", "content": prompt_text},
                    ],
                    "max_tokens": max_tok,
                    "temperature": 0.2,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            ]
        }
        t0 = time.time()
        pred_resp = api_request("POST", f"{base_ep_url}:predict", body=predict_payload, timeout=180)
        dt = time.time() - t0
        pred_obj = pred_resp.get("predictions", [{}])[0]
        usage = pred_obj.get("usage", {}) if isinstance(pred_obj, dict) else {}
        comp_tokens = usage.get("completion_tokens", 0)
        tps = comp_tokens / dt if dt > 0 else 0.0
        print(f"  -> `:predict` succeeded in {dt:.3f}s ({comp_tokens} tokens, {tps:.2f} tok/s)", flush=True)
        print(f"  -> Response preview: {json.dumps(pred_obj, ensure_ascii=False)[:400]}", flush=True)
        results.append({"test": title, "method": ":predict", "latency_s": round(dt, 3), "tokens": comp_tokens, "tok_per_s": round(tps, 2)})

    print(f"\n[Test 4: Multilingual Generation (:rawPredict)] Prompt: {custom_prompt!r}...", flush=True)
    raw_payload = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": custom_prompt}],
        "max_tokens": 1024,
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t1 = time.time()
    raw_resp = api_request("POST", f"{base_ep_url}:rawPredict", body=raw_payload, timeout=180)
    t_raw = time.time() - t1

    choices = raw_resp.get("choices", [{}])
    content = choices[0].get("message", {}).get("content", "")
    usage = raw_resp.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    tok_per_s = comp_tokens / t_raw if t_raw > 0 else 0.0

    print(f"  -> `:rawPredict` succeeded in {t_raw:.3f}s ({comp_tokens} completion tokens, {tok_per_s:.2f} tok/s)", flush=True)
    print("-" * 60)
    print(content.strip())
    print("-" * 60)
    results.append({"test": "Test 4: Multilingual Poem (:rawPredict)", "method": ":rawPredict", "latency_s": round(t_raw, 3), "tokens": comp_tokens, "tok_per_s": round(tok_per_s, 2)})
    return results


# ==============================================================================
# Stage 5: Multi-Concurrency Stress Benchmark
# ==============================================================================

def stage5_stress_benchmark(
    project_id: str,
    region: str,
    endpoint_id: str,
    served_model_name: str,
    concurrency_levels: List[int],
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"STAGE 5: Running Multi-Concurrency Performance Benchmark ({concurrency_levels})")
    print("=" * 80)

    url = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{region}/endpoints/{endpoint_id}:rawPredict"
    )
    prompts = [
        "Explain how Mixture-of-Experts routing works in transformer architectures in 5 concise bullet points.",
        "Write a Python function that computes top-k softmax gating with load balancing loss.",
        "Compare tensor parallelism and expert parallelism for serving large MoE models on 4 GPUs.",
        "Explain how CPU offloading of N-gram Engram embedding tables saves GPU VRAM on g4-standard-192.",
        "Explain why FP8 KV-cache quantization doubles effective context capacity on Blackwell GPUs.",
        "Write a Python async worker pool using asyncio.Queue that limits concurrency to N tasks.",
    ]

    def single_request(idx: int) -> Dict[str, Any]:
        payload = {
            "model": served_model_name,
            "messages": [{"role": "user", "content": prompts[idx % len(prompts)]}],
            "max_tokens": 128,
            "temperature": 0.4,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t_start = time.time()
        body = api_request("POST", url, body=payload, timeout=180)
        latency = time.time() - t_start
        usage = body.get("usage", {})
        return {
            "latency_s": latency,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    summary: Dict[str, Any] = {}
    for c in concurrency_levels:
        total_reqs = max(4, c * 2)
        print(f"\nRunning concurrency={c} ({total_reqs} total requests)...", flush=True)
        t_wall_start = time.time()
        results = []
        errors = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
            futs = [pool.submit(single_request, i) for i in range(total_reqs)]
            for f in concurrent.futures.as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as exc:
                    errors += 1
                    print(f"  [WARN] Request error: {exc}", flush=True)
        wall_s = time.time() - t_wall_start
        latencies = sorted(r["latency_s"] for r in results)
        total_out_tokens = sum(r["completion_tokens"] for r in results)
        agg_tps = total_out_tokens / wall_s if wall_s > 0 else 0.0
        rps = len(results) / wall_s if wall_s > 0 else 0.0

        def pct(arr: List[float], p: float) -> float:
            if not arr:
                return 0.0
            return arr[min(len(arr) - 1, int(len(arr) * p / 100.0))]

        p50 = pct(latencies, 50)
        p90 = pct(latencies, 90)
        p95 = pct(latencies, 95)
        p99 = pct(latencies, 99)
        mean_lat = statistics.mean(latencies) if latencies else 0.0

        summary[str(c)] = {
            "concurrency": c,
            "requests": total_reqs,
            "succeeded": len(results),
            "errors": errors,
            "wall_time_s": round(wall_s, 3),
            "rps": round(rps, 2),
            "p50_latency_s": round(p50, 3),
            "p90_latency_s": round(p90, 3),
            "p95_latency_s": round(p95, 3),
            "p99_latency_s": round(p99, 3),
            "mean_latency_s": round(mean_lat, 3),
            "total_output_tokens": total_out_tokens,
            "aggregate_output_tok_per_s": round(agg_tps, 2),
        }
        print(
            f"  -> Concurrency {c}: {len(results)}/{total_reqs} OK | RPS={rps:.2f} | "
            f"p50={p50:.2f}s | p95={p95:.2f}s | p99={p99:.2f}s | Aggregate={agg_tps:.2f} tok/s",
            flush=True,
        )

    return summary


# ==============================================================================
# Configuration Loader & CLI Argument Parser
# ==============================================================================

def load_config_file(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-Command End-to-End Vertex AI Deployment for Qwen/Qwen3.8-Flash-Next on a Reserved 4-GPU G4 Node in me-central2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to JSON configuration file containing all deployment parameters.",
    )
    parser.add_argument("--hf-model", default=None, help="Override model.hf_model (repo ID or full HF URL).")
    parser.add_argument("--hf-token", default=None, help="Override model.hf_token.")
    parser.add_argument("--model-name", default=None, help="Override model.model_name.")
    parser.add_argument("--served-model-name", default=None, help="Override model.served_model_name.")
    parser.add_argument("--source-gcs-uri", default=None, help="Optional source gs://bucket/prefix if copying weights from an existing GCS bucket.")
    parser.add_argument("--project-id", default=None, help="Override gcp.project_id.")
    parser.add_argument("--region", default=None, help="Override gcp.regions (default: me-central2).")
    parser.add_argument("--bucket-name", default=None, help="Override gcp.bucket_name.")
    parser.add_argument("--gcs-prefix", default=None, help="Override gcp.gcs_prefix.")
    parser.add_argument("--transfer-mode", choices=["auto", "cloudbuild", "local"], default=None, help="Override gcp.transfer_mode.")
    parser.add_argument("--transfer-workers", type=int, default=None, help="Override gcp.transfer_workers.")
    parser.add_argument("--ar-repo", default=None, help="Override gcp.ar_repo.")
    parser.add_argument("--image-uri", default=None, help="Override gcp.image_uri.")
    parser.add_argument("--local-docker-build", action="store_true", help="Build and push container using local Docker instead of Cloud Build.")
    parser.add_argument(
        "--machine-type",
        choices=["g4-standard-48", "g4-standard-96", "g4-standard-192", "g4-standard-384"],
        default=None,
        help="Override compute.machine_type (default: g4-standard-192 for 4 GPUs).",
    )
    parser.add_argument("--accelerator-type", default=None, help="Override compute.accelerator_type.")
    parser.add_argument("--gpu-count", type=int, default=None, help="Override compute.gpu_count (default: 4).")
    parser.add_argument(
        "--capacity-type",
        choices=["reserved", "standard", "spot"],
        default=None,
        help="Override compute.capacity_type ('reserved', 'standard', or 'spot').",
    )
    parser.add_argument(
        "--reservation-affinity-type",
        choices=["SPECIFIC_RESERVATION", "ANY_RESERVATION", "NO_RESERVATION"],
        default=None,
        help="Override compute.reservation.affinity_type.",
    )
    parser.add_argument("--reservation-name", default=None, help="Override compute.reservation.name (short name or full resource path, or AUTO_DISCOVER).")
    parser.add_argument("--reservation-zone", default=None, help="Override compute.reservation.zone (default: me-central2-a).")
    parser.add_argument("--reservation-project-id", default=None, help="Override compute.reservation.project_id (for Shared Reservations).")
    parser.add_argument("--tp-size", type=int, default=None, help="Override vllm.tensor_parallel_size (default: 4).")
    parser.add_argument("--engram-cpu-offload", choices=["auto", "true", "false"], default=None, help="Override vllm.engram_cpu_offload.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override vllm.max_model_len.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="Override vllm.gpu_memory_utilization.")
    parser.add_argument("--kv-cache-dtype", default=None, help="Override vllm.kv_cache_dtype.")
    parser.add_argument("--skip-hf-transfer", action="store_true", help="Skip Stage 1 (weight staging to GCS).")
    parser.add_argument("--skip-container-build", action="store_true", help="Skip Stage 2 (custom vLLM container build).")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip Stage 3 (Vertex AI model upload & endpoint deploy).")
    parser.add_argument("--endpoint-id", default=None, help="Override pipeline_stages.existing_endpoint_id.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip Stage 4 verification requests.")
    parser.add_argument("--prompt", default=None, help="Override pipeline_stages.verification_prompt.")
    parser.add_argument("--run-stress-test", action="store_true", help="Enable Stage 5 concurrent load benchmark.")
    parser.add_argument("--skip-stress-test", action="store_true", help="Skip Stage 5 concurrent load benchmark.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config_file(Path(args.config))

    m_cfg = cfg.get("model", {})
    g_cfg = cfg.get("gcp", {})
    c_cfg = cfg.get("compute", {})
    r_cfg = c_cfg.get("reservation", {})
    v_cfg = cfg.get("vllm", {})
    p_cfg = cfg.get("pipeline_stages", {})

    raw_hf_model = args.hf_model or m_cfg.get("hf_model", "Qwen/Qwen3.8-Flash-Next")
    hf_model_id = normalize_hf_model_id(raw_hf_model)
    hf_token = args.hf_token if args.hf_token is not None else (m_cfg.get("hf_token") or os.environ.get("HF_TOKEN", ""))
    model_name = args.model_name or m_cfg.get("model_name", "qwen3-8-flash-next")
    served_model_name = args.served_model_name or m_cfg.get("served_model_name") or hf_model_id
    local_weights_dir = SCRIPT_DIR / (m_cfg.get("local_weights_dir") or "model_weights")
    source_gcs_uri = args.source_gcs_uri or m_cfg.get("source_gcs_uri", "")

    raw_project_id = args.project_id or os.environ.get("PROJECT_ID") or g_cfg.get("project_id", "")
    if not raw_project_id or raw_project_id == "YOUR_GCP_PROJECT_ID":
        raw_project_id = run_cmd(["gcloud", "config", "get-value", "project"], check=False).stdout.strip()
    if not raw_project_id or raw_project_id in ("(unset)", "YOUR_GCP_PROJECT_ID"):
        sys.exit("ERROR: Set gcp.project_id in deployment_config.json or pass --project-id <YOUR_GCP_PROJECT_ID>.")
    project_id = raw_project_id

    if args.region:
        candidate_regions = [r.strip() for r in args.region.split(",") if r.strip()]
    else:
        cfg_regions = g_cfg.get("regions", ["me-central2"])
        candidate_regions = cfg_regions if isinstance(cfg_regions, list) else [r.strip() for r in str(cfg_regions).split(",")]
    primary_region = candidate_regions[0]

    raw_bucket = args.bucket_name or os.environ.get("BUCKET_NAME") or g_cfg.get("bucket_name", "")
    if not raw_bucket or raw_bucket == "YOUR_GCS_BUCKET_NAME":
        raw_bucket = f"{project_id}-qwen3-8-flash-next-me-central2"
        print(f"[Config] bucket_name was placeholder; defaulting to: gs://{raw_bucket}", flush=True)
    bucket_name = raw_bucket

    gcs_prefix = (args.gcs_prefix or g_cfg.get("gcs_prefix", "vertex_model_ready")).strip("/")
    transfer_mode = args.transfer_mode or g_cfg.get("transfer_mode", "auto")
    transfer_workers = args.transfer_workers or int(g_cfg.get("transfer_workers", 4))
    ar_repo = args.ar_repo or g_cfg.get("ar_repo", "vllm-repo")

    cfg_image_uri = g_cfg.get("image_uri", "").replace("YOUR_GCP_PROJECT_ID", project_id)
    image_uri = (
        args.image_uri
        or cfg_image_uri
        or f"{primary_region}-docker.pkg.dev/{project_id}/{ar_repo}/vllm-qwen38:v5"
    )

    machine_type = args.machine_type or c_cfg.get("machine_type", "g4-standard-192")
    accelerator_type = args.accelerator_type or c_cfg.get("accelerator_type", "NVIDIA_RTX_PRO_6000")
    gpu_count = args.gpu_count or G4_MACHINE_GPU_MAP.get(machine_type) or int(c_cfg.get("gpu_count", 4))
    capacity_type = (args.capacity_type or c_cfg.get("capacity_type", "reserved")).lower()

    res_affinity_type = args.reservation_affinity_type or r_cfg.get("affinity_type", "SPECIFIC_RESERVATION")
    res_name = args.reservation_name or os.environ.get("RESERVATION_NAME") or r_cfg.get("name", "YOUR_RESERVATION_NAME")
    res_zone = args.reservation_zone or os.environ.get("RESERVATION_ZONE") or r_cfg.get("zone", f"{primary_region}-a")
    res_project_id = (
        args.reservation_project_id
        or os.environ.get("RESERVATION_PROJECT_ID")
        or r_cfg.get("project_id", project_id)
    )
    if res_project_id == "YOUR_GCP_PROJECT_ID":
        res_project_id = project_id

    min_replicas = int(c_cfg.get("min_replicas", 1))
    max_replicas = int(c_cfg.get("max_replicas", 1))
    shared_memory_size_mb = int(c_cfg.get("shared_memory_size_mb", 65536))
    deployment_timeout_seconds = int(c_cfg.get("deployment_timeout_seconds", 3600))

    tp_size = args.tp_size or (gpu_count if args.machine_type else int(v_cfg.get("tensor_parallel_size", gpu_count)))
    enable_expert_parallel = bool(v_cfg.get("enable_expert_parallel", True))
    engram_setting = args.engram_cpu_offload or str(v_cfg.get("engram_cpu_offload", "true")).lower()
    if engram_setting == "auto":
        engram_cpu_offload = gpu_count <= 4
    else:
        engram_cpu_offload = engram_setting == "true"

    max_model_len = args.max_model_len or int(v_cfg.get("max_model_len", 32768))
    gpu_memory_utilization = args.gpu_memory_utilization or float(v_cfg.get("gpu_memory_utilization", 0.90))
    kv_cache_dtype = args.kv_cache_dtype or v_cfg.get("kv_cache_dtype", "fp8")
    reasoning_parser = v_cfg.get("reasoning_parser", "qwen3")
    tool_call_parser = v_cfg.get("tool_call_parser", "qwen3_coder")
    enable_auto_tool_choice = bool(v_cfg.get("enable_auto_tool_choice", True))
    trust_remote_code = bool(v_cfg.get("trust_remote_code", True))

    run_hf_transfer = bool(p_cfg.get("run_hf_transfer", True)) and not args.skip_hf_transfer
    run_container_build = bool(p_cfg.get("run_container_build", True)) and not args.skip_container_build
    run_vertex_deploy = bool(p_cfg.get("run_vertex_deploy", True)) and not args.skip_deploy
    endpoint_id = args.endpoint_id or p_cfg.get("existing_endpoint_id", "")
    run_verification_tests = bool(p_cfg.get("run_verification_tests", True)) and not args.skip_tests
    verification_prompt = args.prompt or p_cfg.get(
        "verification_prompt",
        "fais moi un poeme de 12 vers en arabe et la traduction en anglais",
    )
    run_stress_benchmark = (bool(p_cfg.get("run_stress_benchmark", True)) or args.run_stress_test) and not args.skip_stress_test
    concurrency_levels = [int(x) for x in p_cfg.get("benchmark_concurrency_levels", [1, 8, 16, 32])]

    reservation_affinity = None
    if run_vertex_deploy:
        reservation_affinity = resolve_reservation_affinity(
            project_id=project_id,
            capacity_type=capacity_type,
            affinity_type=res_affinity_type,
            reservation_name=res_name,
            reservation_zone=res_zone,
            reservation_project_id=res_project_id,
            machine_type=machine_type,
        )

    gcs_model_uri = f"gs://{bucket_name}/{gcs_prefix}"
    model_display_name = f"{model_name}-vllm-{gpu_count}gpu"
    endpoint_display_name = f"{model_name}-g4-{gpu_count}gpu-ep"

    print("=" * 80)
    print("VERTEX AI 4-GPU RESERVED G4 DEPLOYMENT CONFIGURATION (me-central2)")
    print("=" * 80)
    print(f"  Config File          : {Path(args.config).resolve()}")
    print(f"  Project ID           : {project_id}")
    print(f"  Hugging Face Model   : {hf_model_id}")
    print(f"  Served Model Name    : {served_model_name}")
    print(f"  Bundled Weights Dir  : {local_weights_dir}")
    print(f"  Target Region        : {', '.join(candidate_regions)}")
    print(f"  GCS Model URI        : {gcs_model_uri}")
    print(f"  Container Image URI  : {image_uri}")
    print(f"  Machine Type         : {machine_type} ({gpu_count}x {accelerator_type})")
    print(f"  Capacity Mode        : {capacity_type.upper()}")
    print(f"  Reservation Affinity : {json.dumps(reservation_affinity) if reservation_affinity else 'None'}")
    print(
        f"  vLLM Configuration   : TP={tp_size}, EP={enable_expert_parallel}, "
        f"engram_cpu_offload={engram_cpu_offload}, max_len={max_model_len}, kv={kv_cache_dtype}"
    )
    print("=" * 80)

    # Stage 1: Transfer model weights to customer's GCS bucket in me-central2
    if run_hf_transfer:
        stage1_transfer_weights_to_gcs(
            project_id=project_id,
            region=primary_region,
            hf_model_id=hf_model_id,
            bucket_name=bucket_name,
            gcs_prefix=gcs_prefix,
            num_workers=transfer_workers,
            local_weights_dir=local_weights_dir,
            source_gcs_uri=source_gcs_uri,
            transfer_mode=transfer_mode,
            hf_token=hf_token,
        )
    else:
        print(f"\n[SKIP Stage 1] Using existing GCS weights at {gcs_model_uri}", flush=True)

    # Stage 2: Build and push custom Vertex AI vLLM container in me-central2
    if run_container_build:
        stage2_build_vllm_container(
            project_id=project_id,
            region=primary_region,
            ar_repo=ar_repo,
            image_uri=image_uri,
            local_docker_build=args.local_docker_build,
        )
    else:
        print(f"\n[SKIP Stage 2] Using existing container image {image_uri}", flush=True)

    # Stage 3: Upload Model, Create Endpoint & Deploy to Reserved 4-GPU G4 Node
    active_region = primary_region
    model_res_name = ""
    deployed_model_id = ""

    if run_vertex_deploy:
        print("\n" + "=" * 80)
        print(f"STAGE 3: Vertex AI Model Upload & Reserved 4-GPU Deployment ({primary_region})")
        print("=" * 80)
        deployed_ok = False
        for reg in candidate_regions:
            try:
                ok, model_res_name, endpoint_id, deployed_model_id = deploy_in_region(
                    project_id=project_id,
                    region=reg,
                    model_display_name=model_display_name,
                    endpoint_display_name=endpoint_display_name,
                    served_model_name=served_model_name,
                    gcs_model_uri=gcs_model_uri,
                    image_uri=image_uri,
                    machine_type=machine_type,
                    accelerator_type=accelerator_type,
                    gpu_count=gpu_count,
                    capacity_type=capacity_type,
                    reservation_affinity=reservation_affinity,
                    min_replicas=min_replicas,
                    max_replicas=max_replicas,
                    shared_memory_size_mb=shared_memory_size_mb,
                    deployment_timeout_seconds=deployment_timeout_seconds,
                    tp_size=tp_size,
                    enable_expert_parallel=enable_expert_parallel,
                    engram_cpu_offload=engram_cpu_offload,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    kv_cache_dtype=kv_cache_dtype,
                    reasoning_parser=reasoning_parser,
                    tool_call_parser=tool_call_parser,
                    enable_auto_tool_choice=enable_auto_tool_choice,
                    trust_remote_code=trust_remote_code,
                )
                if ok:
                    active_region = reg
                    deployed_ok = True
                    break
            except Exception as exc:
                print(f"[{reg}] Exception during deploy: {exc}", flush=True)
            if len(candidate_regions) > 1:
                print(f"[{reg}] Moving to next candidate fallback region...", flush=True)

        if not deployed_ok:
            raise RuntimeError(f"Deployment failed across candidate regions: {candidate_regions}")
    else:
        if not endpoint_id:
            sys.exit("ERROR: When Stage 3 deploy is skipped, provide existing_endpoint_id in deployment_config.json or --endpoint-id.")
        print(f"\n[SKIP Stage 3] Using existing Vertex AI Endpoint {endpoint_id} in {active_region}", flush=True)

    # Stage 4: Verify :predict & :rawPredict
    e2e_results = []
    if run_verification_tests:
        e2e_results = stage4_verify_endpoint(
            project_id=project_id,
            region=active_region,
            endpoint_id=endpoint_id,
            served_model_name=served_model_name,
            custom_prompt=verification_prompt,
        )

    # Stage 5: Multi-Concurrency Stress Benchmark
    if run_stress_benchmark:
        bench_results = stage5_stress_benchmark(
            project_id=project_id,
            region=active_region,
            endpoint_id=endpoint_id,
            served_model_name=served_model_name,
            concurrency_levels=concurrency_levels,
        )
        out_path = SCRIPT_DIR / "scripts" / "stress_test_results_latest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "project_id": project_id,
                    "region": active_region,
                    "endpoint_id": endpoint_id,
                    "machine_type": machine_type,
                    "gpu_count": gpu_count,
                    "reservation_affinity": reservation_affinity,
                    "e2e_verification": e2e_results,
                    "stress_benchmark": bench_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved E2E verification and stress benchmark results to {out_path}", flush=True)

    print("\n" + "=" * 80)
    print("END-TO-END PIPELINE COMPLETE")
    print("=" * 80)
    print(f"  Project ID        : {project_id}")
    print(f"  Active Region     : {active_region}")
    if model_res_name:
        print(f"  Vertex Model      : {model_res_name}")
    print(f"  Endpoint ID       : {endpoint_id}")
    if deployed_model_id:
        print(f"  Deployed Model ID : {deployed_model_id}")
    print(
        f"  Predict URL       : https://{active_region}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{active_region}/endpoints/{endpoint_id}:predict"
    )
    print(
        f"  RawPredict URL    : https://{active_region}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{active_region}/endpoints/{endpoint_id}:rawPredict"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
