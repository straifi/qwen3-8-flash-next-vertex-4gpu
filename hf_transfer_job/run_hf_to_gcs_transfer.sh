#!/usr/bin/env bash
# End-to-end script to create the customer's GCS bucket in me-central2,
# upload the bundled model_weights/ config/tokenizer/index files from this GitHub repo,
# and launch 4 parallel Cloud Build E2_HIGHCPU_32 jobs that stream all 131 .safetensors
# shards (335.3 GiB) of Qwen/Qwen3.8-Flash-Next into gs://${BUCKET_NAME}/vertex_model_ready/.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-me-central2}"
BUCKET_NAME="${BUCKET_NAME:-YOUR_GCS_BUCKET_NAME}"
GCS_PREFIX="${GCS_PREFIX:-vertex_model_ready}"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "YOUR_GCP_PROJECT_ID" ]]; then
  echo "ERROR: Set PROJECT_ID=<YOUR_GCP_PROJECT_ID> before running this script."
  exit 1
fi
if [[ -z "${BUCKET_NAME}" || "${BUCKET_NAME}" == "YOUR_GCS_BUCKET_NAME" ]]; then
  echo "ERROR: Set BUCKET_NAME=<YOUR_GCS_BUCKET_NAME> before running this script."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

echo "[1/4] Ensuring GCS bucket gs://${BUCKET_NAME} exists in ${REGION}..."
gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access

echo "[2/4] Uploading bundled model_weights/ config, tokenizer & shard-index files from GitHub repo to gs://${BUCKET_NAME}/${GCS_PREFIX}/..."
gcloud storage cp \
  "${REPO_ROOT}/model_weights/LICENSE" \
  "${REPO_ROOT}/model_weights/README.md" \
  "${REPO_ROOT}/model_weights/chat_template.jinja" \
  "${REPO_ROOT}/model_weights/config.json" \
  "${REPO_ROOT}/model_weights/generation_config.json" \
  "${REPO_ROOT}/model_weights/merges.txt" \
  "${REPO_ROOT}/model_weights/model.safetensors.index.json" \
  "${REPO_ROOT}/model_weights/preprocessor_config.json" \
  "${REPO_ROOT}/model_weights/tokenizer.json" \
  "${REPO_ROOT}/model_weights/tokenizer_config.json" \
  "${REPO_ROOT}/model_weights/video_preprocessor_config.json" \
  "${REPO_ROOT}/model_weights/vocab.json" \
  "gs://${BUCKET_NAME}/${GCS_PREFIX}/"

echo "[3/4] Submitting 4 parallel Cloud Build E2_HIGHCPU_32 jobs to stream shards 1..131 (335.3 GiB)..."
gcloud builds submit . --project="${PROJECT_ID}" --config=cloudbuild.yaml --async \
  --substitutions="_BUCKET_NAME=${BUCKET_NAME},_GCS_PREFIX=${GCS_PREFIX},_START_SHARD=1,_END_SHARD=33,_TOTAL_SHARDS=131"

gcloud builds submit . --project="${PROJECT_ID}" --config=cloudbuild.yaml --async \
  --substitutions="_BUCKET_NAME=${BUCKET_NAME},_GCS_PREFIX=${GCS_PREFIX},_START_SHARD=34,_END_SHARD=66,_TOTAL_SHARDS=131"

gcloud builds submit . --project="${PROJECT_ID}" --config=cloudbuild.yaml --async \
  --substitutions="_BUCKET_NAME=${BUCKET_NAME},_GCS_PREFIX=${GCS_PREFIX},_START_SHARD=67,_END_SHARD=99,_TOTAL_SHARDS=131"

gcloud builds submit . --project="${PROJECT_ID}" --config=cloudbuild.yaml \
  --substitutions="_BUCKET_NAME=${BUCKET_NAME},_GCS_PREFIX=${GCS_PREFIX},_START_SHARD=100,_END_SHARD=131,_TOTAL_SHARDS=131"

echo "[4/4] Verifying staged files in gs://${BUCKET_NAME}/${GCS_PREFIX}/..."
gcloud storage ls "gs://${BUCKET_NAME}/${GCS_PREFIX}/model-*.safetensors" | wc -l
gcloud storage du -sh "gs://${BUCKET_NAME}/${GCS_PREFIX}"
