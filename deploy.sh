#!/usr/bin/env bash
# ==============================================================================
# One-Command Customer Deployment & Benchmark Entrypoint for Qwen3.8-Flash-Next
# on a Reserved 4-RTX G4 Node (`g4-standard-192`) in `me-central2` (`me-central2-a`).
#
# Usage (Option A - Edit deployment_config.json first):
#   ./deploy.sh
#
# Usage (Option B - Pass customer parameters directly on the command line):
#   ./deploy.sh \
#     --project-id YOUR_GCP_PROJECT_ID \
#     --bucket-name YOUR_GCS_BUCKET_NAME \
#     --reservation-name YOUR_RESERVATION_NAME
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "================================================================================"
echo "[Pre-Flight] Checking Python dependencies and Google Cloud CLI..."
echo "================================================================================"
if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: 'gcloud' CLI is required. Install Google Cloud SDK and run 'gcloud auth login'."
  exit 1
fi

python3 -c "import google.auth, huggingface_hub, google.cloud.storage" >/dev/null 2>&1 || {
  echo "[Pre-Flight] Installing required Python packages (google-auth, requests, huggingface_hub, hf_transfer, google-cloud-storage)..."
  pip3 install --quiet --user google-auth requests huggingface_hub hf_transfer google-cloud-storage || \
  pip3 install --quiet --break-system-packages google-auth requests huggingface_hub hf_transfer google-cloud-storage
}

exec python3 -u "${SCRIPT_DIR}/deploy_end_to_end.py" "$@"
