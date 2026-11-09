#!/usr/bin/env python3
"""
Standalone Stage 3 Deployer & Resumable LRO Monitor for Qwen/Qwen3.8-Flash-Next
on a Reserved 4-GPU G4 Node (`g4-standard-192`, 4x NVIDIA RTX PRO 6000) in `me-central2` (`me-central2-a`):
1. Reads configuration from `deployment_config.json` (or CLI overrides).
2. Uploads the model to Vertex AI Model Registry in `me-central2` with `TP=4`,
   `--enable-expert-parallel`, and `--engram-config={"cpu_offload":true}`.
3. Deploys the model to the Vertex AI Endpoint in `me-central2` pinned to the customer's
   specific Compute Engine reservation in `me-central2-a`.
4. Immediately runs `scripts/stress_test_endpoint.py` once deployment succeeds.
"""
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if __name__ == "__main__":
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "deploy_end_to_end.py"),
        "--skip-hf-transfer",
        "--skip-container-build",
    ] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
