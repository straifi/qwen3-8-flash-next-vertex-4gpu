#!/usr/bin/env python3
"""
Stages all 143 model files for Qwen/Qwen3.8-Flash-Next into the customer's
Google Cloud Storage bucket in `me-central2` (`gs://<BUCKET_NAME>/vertex_model_ready/`):
1. Uploads the 12 model config/tokenizer/index files directly from `./model_weights/`
   bundled inside this GitHub repository (if START_SHARD == 1).
2. Streams `.safetensors` weight shards (`model-00001-of-00131.safetensors` ..
   `model-00131-of-00131.safetensors`) in bounded batches of 4 using Rust-accelerated
   `hf_transfer` (`HF_HUB_ENABLE_HF_TRANSFER=1`), uploads each batch to GCS in parallel,
   and deletes local files between batches so scratch disks never fill up.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from google.cloud import storage

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

REPO_ID = os.environ.get("REPO_ID", "Qwen/Qwen3.8-Flash-Next")
PROJECT_ID = os.environ.get("PROJECT_ID") or None
BUCKET_NAME = os.environ.get("BUCKET_NAME", "YOUR_GCS_BUCKET_NAME")
GCS_PREFIX = os.environ.get("GCS_PREFIX", "vertex_model_ready").strip("/")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

START_SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("START_SHARD", "1"))
END_SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("END_SHARD", "131"))
TOTAL_SHARDS = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("TOTAL_SHARDS", "131"))

storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)
gs_dest = f"gs://{BUCKET_NAME}/{GCS_PREFIX}/"

tmp_dir = "/workspace/hf_tmp" if os.path.isdir("/workspace") else "/tmp/hf_tmp"
os.makedirs(tmp_dir, exist_ok=True)

script_dir = Path(__file__).resolve().parent
repo_weights_dir = script_dir.parent / "model_weights"

# 1. If starting at shard 1, stage the 12 config/tokenizer/index files
if START_SHARD == 1:
    local_bundled = []
    if repo_weights_dir.is_dir():
        for p in sorted(repo_weights_dir.iterdir()):
            if p.is_file() and p.name not in ("shard_manifest.json", "stage_weights_to_gcs.py") and not p.name.endswith(".py"):
                local_bundled.append(str(p))

    if local_bundled:
        print(f"Uploading {len(local_bundled)} bundled model config/tokenizer files from {repo_weights_dir} to {gs_dest}...", flush=True)
        subprocess.run(["gcloud", "storage", "cp"] + local_bundled + [gs_dest], check=True)
    else:
        api = HfApi(token=HF_TOKEN)
        config_files = [
            f for f in api.list_repo_files(REPO_ID)
            if not f.endswith(".safetensors") and not f.startswith(".")
        ]
        print(f"Downloading {len(config_files)} config/tokenizer files from {REPO_ID}: {config_files}", flush=True)
        cfg_dir = os.path.join(tmp_dir, "configs")
        os.makedirs(cfg_dir, exist_ok=True)
        local_cfgs = [hf_hub_download(repo_id=REPO_ID, filename=f, local_dir=cfg_dir, token=HF_TOKEN) for f in config_files]
        subprocess.run(["gcloud", "storage", "cp"] + local_cfgs + [gs_dest], check=True)
        shutil.rmtree(cfg_dir, ignore_errors=True)
        print(f"Uploaded {len(config_files)} config/tokenizer files to {gs_dest}", flush=True)

# 2. Check existing blobs in GCS to skip already-uploaded shards
existing = set()
for blob in bucket.list_blobs(prefix=f"{GCS_PREFIX}/model-"):
    if blob.size and blob.size > 1_000_000:
        existing.add(blob.name.split("/")[-1])

shards = [
    f"model-{i:05d}-of-{TOTAL_SHARDS:05d}.safetensors"
    for i in range(START_SHARD, END_SHARD + 1)
    if f"model-{i:05d}-of-{TOTAL_SHARDS:05d}.safetensors" not in existing
]

print(
    f"Transferring {len(shards)} shards ({START_SHARD}..{END_SHARD} of {TOTAL_SHARDS}) "
    f"from {REPO_ID} to {gs_dest}",
    flush=True,
)

BATCH_SIZE = 4
for b_idx in range(0, len(shards), BATCH_SIZE):
    batch = shards[b_idx:b_idx + BATCH_SIZE]
    print(f"--- Batch {b_idx // BATCH_SIZE + 1}: {batch} ---", flush=True)

    local_files = []
    for fname in batch:
        print(f"Downloading {fname}...", flush=True)
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            local_dir=tmp_dir,
            token=HF_TOKEN,
        )
        local_files.append(path)

    cmd = ["gcloud", "storage", "cp"] + local_files + [gs_dest]
    print(f"Uploading batch to {gs_dest}...", flush=True)
    subprocess.run(cmd, check=True)

    for path in local_files:
        try:
            os.remove(path)
        except OSError:
            pass
    shutil.rmtree(os.path.join(tmp_dir, ".cache"), ignore_errors=True)

print(f"Successfully transferred shards {START_SHARD}..{END_SHARD}!", flush=True)
