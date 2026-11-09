#!/usr/bin/env python3
"""
Resumable LRO Deployment Monitor & Automated E2E + Stress-Test Runner for `me-central2`:
1. Reads active deployment state from `scripts/active_deployment.json` (or `deployment_config.json`).
2. Polls the active Vertex AI `deployModel` Long-Running Operation (LRO) and streams live
   container logs from Cloud Logging.
3. Automatically launches `scripts/stress_test_endpoint.py` as soon as the endpoint is ONLINE.
"""
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
import google.auth
import google.auth.transport.requests

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = REPO_ROOT / "deployment_config.json"
STATE_FILE = SCRIPT_DIR / "active_deployment.json"
STRESS_SCRIPT = SCRIPT_DIR / "stress_test_endpoint.py"
STRESS_OUT = SCRIPT_DIR / "stress_test_results.json"


def get_token() -> str:
    try:
        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        if creds.token:
            return creds.token
    except Exception:
        pass
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def api_request(method, url, body=None, timeout=60):
    token = get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "vertex-ai-deployment-monitor/1.0",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def monitor_operation(project_id: str, region: str, ep_id: str, model_name: str, op_name: str, machine_type: str, accel_count: int) -> bool:
    seen_ts = set()
    start_iso = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    while True:
        try:
            op = api_request("GET", f"https://{region}-aiplatform.googleapis.com/v1/{op_name}")
            done = op.get("done", False)
            stage = op.get("metadata", {}).get("deploymentStage", "UNKNOWN")
            now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%SZ")
            print(f"[{now_str}][{region}][{machine_type}/{accel_count}xGPU] done={done}, stage={stage}", flush=True)

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
            entries = list(reversed(logs.get("entries", [])))
            for e in entries:
                ts = e.get("timestamp", "")
                msg = e.get("textPayload") or json.dumps(e.get("jsonPayload", {}))
                key = (ts, msg[:120])
                if key not in seen_ts:
                    seen_ts.add(key)
                    print(f"  [{ts}] {msg.strip()[:280]}", flush=True)

            if done:
                if "error" in op:
                    print(f"[{region}] DEPLOYMENT ERROR: {json.dumps(op['error'], indent=2)}", flush=True)
                    return False

                deployed_model_id = op.get("response", {}).get("deployedModel", {}).get("id")
                print(
                    f"[{region}] DEPLOYMENT SUCCEEDED! endpoint_id={ep_id}, deployedModelId={deployed_model_id}",
                    flush=True,
                )
                STATE_FILE.write_text(
                    json.dumps(
                        {
                            "project_id": project_id,
                            "region": region,
                            "endpoint_id": ep_id,
                            "model_name": model_name,
                            "deployed_model_id": deployed_model_id,
                            "operation_name": op_name,
                            "machine_type": machine_type,
                            "accelerator_count": accel_count,
                            "status": "DEPLOYED",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"[{region}] Launching E2E & Stress Test Suite against endpoint {ep_id}...", flush=True)
                subprocess.check_call([sys.executable, str(STRESS_SCRIPT), ep_id, region, str(STRESS_OUT)])
                return True
        except Exception as e:
            print(f"  [{region} monitor warning] {e}", flush=True)

        time.sleep(25)


def main():
    if not STATE_FILE.exists():
        print("No active_deployment.json found; invoking deploy_me_central2_reserved.py...", flush=True)
        sys.exit(subprocess.call([sys.executable, str(SCRIPT_DIR / "deploy_me_central2_reserved.py")] + sys.argv[1:]))

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    project_id = state.get("project_id") or os.environ.get("PROJECT_ID", "")
    reg = state.get("region", "me-central2")
    ep_id = state["endpoint_id"]
    model_name = state.get("model_name", "")
    op_name = state["operation_name"]
    mt = state.get("machine_type", "g4-standard-192")
    ac = int(state.get("accelerator_count", 4))
    print(f"Resuming monitor for active deployment in {reg} (op={op_name})...", flush=True)
    monitor_operation(project_id, reg, ep_id, model_name, op_name, mt, ac)


if __name__ == "__main__":
    main()
