#!/usr/bin/env python3
"""
Standalone CLI to stage the bundled Qwen/Qwen3.8-Flash-Next artifacts from `./model_weights/`
plus all 131 `.safetensors` shards (335.30 GiB) into the customer's GCS bucket in `me-central2`.

Usage:
  python3 model_weights/stage_weights_to_gcs.py \
    --project-id YOUR_GCP_PROJECT_ID \
    --bucket-name YOUR_GCS_BUCKET_NAME \
    --region me-central2
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deploy_end_to_end import load_config_file, normalize_hf_model_id, stage1_transfer_weights_to_gcs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage Qwen3.8-Flash-Next model weights to Customer GCS in me-central2")
    parser.add_argument("--config", default=str(REPO_ROOT / "deployment_config.json"))
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--bucket-name", default=None)
    parser.add_argument("--gcs-prefix", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--transfer-mode", choices=["auto", "cloudbuild", "local"], default=None)
    parser.add_argument("--source-gcs-uri", default=None)
    args = parser.parse_args()

    cfg = load_config_file(Path(args.config))
    m_cfg = cfg.get("model", {})
    g_cfg = cfg.get("gcp", {})

    project_id = args.project_id or g_cfg.get("project_id", "YOUR_GCP_PROJECT_ID")
    region = args.region or (g_cfg.get("regions") or ["me-central2"])[0]
    bucket_name = args.bucket_name or g_cfg.get("bucket_name", "YOUR_GCS_BUCKET_NAME")
    gcs_prefix = (args.gcs_prefix or g_cfg.get("gcs_prefix", "vertex_model_ready")).strip("/")
    workers = args.workers or int(g_cfg.get("transfer_workers", 4))
    transfer_mode = args.transfer_mode or g_cfg.get("transfer_mode", "auto")
    source_gcs_uri = args.source_gcs_uri or m_cfg.get("source_gcs_uri", "")
    hf_model_id = normalize_hf_model_id(m_cfg.get("hf_model", "Qwen/Qwen3.8-Flash-Next"))

    stage1_transfer_weights_to_gcs(
        project_id=project_id,
        region=region,
        hf_model_id=hf_model_id,
        bucket_name=bucket_name,
        gcs_prefix=gcs_prefix,
        num_workers=workers,
        local_weights_dir=Path(__file__).resolve().parent,
        source_gcs_uri=source_gcs_uri,
        transfer_mode=transfer_mode,
        hf_token=m_cfg.get("hf_token", ""),
    )


if __name__ == "__main__":
    main()
