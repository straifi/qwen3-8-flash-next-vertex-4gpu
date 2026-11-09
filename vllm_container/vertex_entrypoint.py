#!/usr/bin/env python3
"""
Vertex AI compatible entrypoint wrapper for vLLM (vllm/vllm-openai:cu130-nightly)
optimized for 4-GPU G4 nodes (`g4-standard-192`, 4x NVIDIA RTX PRO 6000) in `me-central2`:
1. Stages model weights from GCS (`gs://...`):
   - Uses 8-thread parallel HTTP Range requests (`os.pwrite`) for high-throughput GCS downloads.
   - If `/dev/shm` or `/tmp` has >= 350 GiB free, downloads all 143 files directly.
   - Otherwise (e.g. 64 GiB `/dev/shm` and ~100 GiB `/tmp`), stages the 12 config/tokenizer
     files (23 MB) into `/tmp/qwen3_model`, creates 1-byte `.safetensors` placeholders, and
     launches a background sliding-window RAM prefetcher into `/dev/shm/qwen3_shards` paired
     with `sitecustomize.py` so all 4 Tensor-Parallel workers (`TP=4`) stream weights at
     RAM speed while keeping >= 22 GiB of `/dev/shm` free for NCCL & PyTorch IPC.
2. Uses the official `vllm serve` CLI entrypoint (`vllm.entrypoints.cli.main:main`)
   so all v0.30.0+ flags (`--engram-config={"cpu_offload":true}`, `--enable-expert-parallel`,
   `--attention-config.indexer_kv_dtype=fp8`, etc.) are supported.
3. Paired with `sitecustomize.py` which patches `fastapi.FastAPI.__init__` so that `/predict`
   is registered on every FastAPI app created by vLLM, supporting both standard Vertex AI
   `:predict` (`{"instances": [...]}`) and `:rawPredict` (OpenAI Chat Completions JSON).
"""
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

SHARD_DIR = "/dev/shm/qwen3_shards"
ACTIVE_FLAG = "/dev/shm/qwen3_stream_active"


def get_gcp_token() -> str:
    """Fetch OAuth2 access token from GCE metadata server."""
    for host in ("169.254.169.254", "metadata.google.internal"):
        url = f"http://{host}/computeMetadata/v1/instance/service-accounts/default/token"
        req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["access_token"]
        except Exception:
            pass
    try:
        import google.auth
        import google.auth.transport.requests
        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        if creds.token:
            return creds.token
    except Exception:
        pass
    return ""


def _download_range_chunk(url: str, fd: int, start_byte: int, end_byte: int, token: str) -> None:
    """Download a byte range of a large GCS object and write directly to file descriptor via os.pwrite."""
    for attempt in range(5):
        try:
            headers = {"Range": f"bytes={start_byte}-{end_byte}"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                offset = start_byte
                while True:
                    chunk = resp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    os.pwrite(fd, chunk, offset)
                    offset += len(chunk)
            return
        except Exception as e:
            if attempt == 4:
                raise RuntimeError(f"Range {start_byte}-{end_byte} failed: {e}") from e
            time.sleep(1.0 * (attempt + 1))


def _download_one_object(task):
    bucket, obj_name, rel_name, expected_size, dest_dir = task
    dest_path = os.path.join(dest_dir, rel_name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) == expected_size:
        return rel_name, expected_size, 0.0

    encoded_obj = urllib.parse.quote(obj_name, safe="")
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{encoded_obj}?alt=media"

    for attempt in range(1, 6):
        t0 = time.time()
        tmp_path = f"{dest_path}.part.{os.getpid()}.{attempt}"
        try:
            token = get_gcp_token()
            if expected_size >= 64 * 1024 * 1024:
                num_threads = 8
                fd = os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
                try:
                    os.ftruncate(fd, expected_size)
                    chunk_size = (expected_size + num_threads - 1) // num_threads
                    ranges = []
                    for i in range(num_threads):
                        s = i * chunk_size
                        e = min(expected_size - 1, (i + 1) * chunk_size - 1)
                        if s <= e:
                            ranges.append((s, e))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as pool:
                        futs = [pool.submit(_download_range_chunk, url, fd, s, e, token) for s, e in ranges]
                        for fut in concurrent.futures.as_completed(futs):
                            fut.result()
                finally:
                    os.close(fd)
                os.replace(tmp_path, dest_path)
                return rel_name, expected_size, time.time() - t0
            else:
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                req = urllib.request.Request(url, headers=headers)
                bytes_written = 0
                with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_written += len(chunk)
                if expected_size > 0 and bytes_written != expected_size:
                    raise IOError(f"Size mismatch for {rel_name}: got {bytes_written}, expected {expected_size}")
                os.replace(tmp_path, dest_path)
                return rel_name, bytes_written, time.time() - t0
        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if attempt == 5:
                raise RuntimeError(f"Failed to download {rel_name} after 5 attempts: {e}") from e
            time.sleep(2 * attempt)


def run_prefetch_daemon(manifest_path: str):
    """Background process that maintains a sliding window of .safetensors shards in /dev/shm/qwen3_shards."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    bucket = manifest["bucket"]
    tp_size = int(manifest.get("tp_size", 4))
    shards = manifest["shards"]  # list of [obj_name, basename, size]
    shard_map = {item[1]: (item[0], item[2]) for item in shards}
    ordered_names = [item[1] for item in shards]

    max_active_shards = 10  # ~26 GiB in 64 GiB /dev/shm, leaving ~32 GiB headroom for NCCL & PyTorch IPC
    in_flight = {}  # basename -> Future
    ready_since = {
        item[1]: time.time()
        for item in shards[:4]
        if os.path.exists(os.path.join(SHARD_DIR, f"{item[1]}.ready"))
    }
    next_idx = len(ready_since)
    completed_count = len(ready_since)
    evicted_count = 0

    print(
        f"[prefetch-daemon] Started for {len(ordered_names)} shards from gs://{bucket} "
        f"(tp_size={tp_size}, max_active_shards={max_active_shards}, pre_staged={completed_count})",
        flush=True,
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

    def _download_shard(bname: str):
        obj_name, expected_size = shard_map[bname]
        ready_name = f"{bname}.ready"
        req_path = os.path.join(SHARD_DIR, f"{bname}.req")
        _, nbytes, dt = _download_one_object((bucket, obj_name, ready_name, expected_size, SHARD_DIR))
        try:
            if os.path.exists(req_path):
                os.remove(req_path)
        except Exception:
            pass
        return bname, nbytes, dt

    try:
        last_health_check = 0.0
        while True:
            # Check if vLLM server is already healthy (all weights loaded)
            if time.time() - last_health_check > 5.0:
                last_health_check = time.time()
                try:
                    with urllib.request.urlopen("http://localhost:8080/health", timeout=1.5) as resp:
                        if resp.status == 200:
                            print("[prefetch-daemon] vLLM /health returned 200! Cleaning up /dev/shm/qwen3_shards...", flush=True)
                            try:
                                if os.path.exists(ACTIVE_FLAG):
                                    os.remove(ACTIVE_FLAG)
                            except Exception:
                                pass
                            shutil.rmtree(SHARD_DIR, ignore_errors=True)
                            return
                except Exception:
                    pass

            # Reap completed downloads
            done_futs = [b for b, fut in list(in_flight.items()) if fut.done()]
            for b in done_futs:
                fut = in_flight.pop(b)
                try:
                    bname, nbytes, dt = fut.result()
                    ready_since[bname] = time.time()
                    completed_count += 1
                    if completed_count <= 10 or completed_count % 10 == 0 or completed_count >= len(ordered_names):
                        print(
                            f"[prefetch-daemon] Ready {bname} ({nbytes / (1024**3):.2f} GiB in {dt:.1f}s) "
                            f"[staged={completed_count}/{len(ordered_names)}, evicted={evicted_count}]",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[prefetch-daemon] Error downloading {b}: {e}", flush=True)

            # Evict shards where all TP workers have finished (done >= tp_size and in_use == 0)
            active_ready = [
                fn[:-len(".ready")]
                for fn in os.listdir(SHARD_DIR)
                if fn.endswith(".ready")
            ]
            for bname in list(active_ready):
                in_use_dir = os.path.join(SHARD_DIR, f"{bname}.in_use")
                done_dir = os.path.join(SHARD_DIR, f"{bname}.done")
                in_use_cnt = len(os.listdir(in_use_dir)) if os.path.isdir(in_use_dir) else 0
                done_cnt = len(os.listdir(done_dir)) if os.path.isdir(done_dir) else 0
                if in_use_cnt == 0 and done_cnt >= tp_size:
                    try:
                        os.remove(os.path.join(SHARD_DIR, f"{bname}.ready"))
                        shutil.rmtree(done_dir, ignore_errors=True)
                        active_ready.remove(bname)
                        ready_since.pop(bname, None)
                        evicted_count += 1
                        if evicted_count <= 5 or evicted_count % 10 == 0 or evicted_count >= len(ordered_names):
                            print(
                                f"[prefetch-daemon] Consumed & evicted {bname} (all {done_cnt} workers done) "
                                f"[evicted={evicted_count}/{len(ordered_names)}]",
                                flush=True,
                            )
                    except Exception:
                        pass

            # Check free space in /dev/shm
            shm_stat = os.statvfs("/dev/shm")
            shm_free_gib = (shm_stat.f_bavail * shm_stat.f_frsize) / (1024 ** 3)
            projected_free_gib = shm_free_gib - (len(in_flight) * 2.8)

            # Check urgent on-demand requests (*.req)
            urgent = []
            for fn in os.listdir(SHARD_DIR):
                if fn.endswith(".req"):
                    bname = fn[:-len(".req")]
                    ready_path = os.path.join(SHARD_DIR, f"{bname}.ready")
                    if bname in shard_map and not os.path.exists(ready_path) and bname not in in_flight:
                        urgent.append(bname)
            urgent.sort()

            # If any worker is urgently waiting for a shard and /dev/shm is full or at window limit,
            # evict non-in-use ready shards so urgent requests never deadlock.
            if urgent and (projected_free_gib < 14.0 or len(active_ready) + len(in_flight) >= max_active_shards):
                urgent_set = set(urgent)
                evict_candidates = []
                for bname in list(active_ready):
                    if bname in urgent_set:
                        continue
                    in_use_dir = os.path.join(SHARD_DIR, f"{bname}.in_use")
                    done_dir = os.path.join(SHARD_DIR, f"{bname}.done")
                    in_use_cnt = len(os.listdir(in_use_dir)) if os.path.isdir(in_use_dir) else 0
                    done_cnt = len(os.listdir(done_dir)) if os.path.isdir(done_dir) else 0
                    req_exists = os.path.exists(os.path.join(SHARD_DIR, f"{bname}.req"))
                    if in_use_cnt == 0 and not req_exists:
                        evict_candidates.append((done_cnt > 0, bname, done_dir))
                evict_candidates.sort(reverse=True)
                for _, evict_b, evict_done in evict_candidates[:len(urgent) + 1]:
                    try:
                        os.remove(os.path.join(SHARD_DIR, f"{evict_b}.ready"))
                        shutil.rmtree(evict_done, ignore_errors=True)
                        active_ready.remove(evict_b)
                        ready_since.pop(evict_b, None)
                        print(f"[prefetch-daemon] Evicted {evict_b} to prioritize urgent request {urgent[0]}", flush=True)
                    except Exception:
                        pass
                shm_stat = os.statvfs("/dev/shm")
                shm_free_gib = (shm_stat.f_bavail * shm_stat.f_frsize) / (1024 ** 3)
                projected_free_gib = shm_free_gib - (len(in_flight) * 2.8)

            # Schedule urgent requests first, then sequential prefetch
            for bname in urgent:
                if len(active_ready) + len(in_flight) < max_active_shards + 2 and projected_free_gib >= 10.0:
                    done_dir = os.path.join(SHARD_DIR, f"{bname}.done")
                    shutil.rmtree(done_dir, ignore_errors=True)
                    os.makedirs(done_dir, exist_ok=True)
                    in_flight[bname] = executor.submit(_download_shard, bname)
                    projected_free_gib -= 2.8

            while (
                next_idx < len(ordered_names)
                and len(active_ready) + len(in_flight) < max_active_shards
                and len(in_flight) < 4
                and projected_free_gib >= 22.0
            ):
                bname = ordered_names[next_idx]
                next_idx += 1
                ready_path = os.path.join(SHARD_DIR, f"{bname}.ready")
                if not os.path.exists(ready_path) and bname not in in_flight:
                    in_flight[bname] = executor.submit(_download_shard, bname)
                    projected_free_gib -= 2.8

            time.sleep(0.1)
    finally:
        executor.shutdown(wait=False)


def stage_gcs_to_local(gcs_uri: str, tp_size: int = 4) -> str:
    """Stage model from gs://bucket/prefix locally or via sliding-window RAM prefetcher."""
    if not gcs_uri.startswith("gs://"):
        if gcs_uri.startswith("/gcs/"):
            if os.path.isdir(gcs_uri) and os.path.exists(os.path.join(gcs_uri, "config.json")):
                print(f"[vertex-vllm-entrypoint] Found existing /gcs mount at {gcs_uri}", flush=True)
                return gcs_uri
            gcs_uri = "gs://" + gcs_uri[len("/gcs/"):]
        else:
            return gcs_uri

    shm_stat = os.statvfs("/dev/shm")
    shm_free_gib = (shm_stat.f_bavail * shm_stat.f_frsize) / (1024 ** 3)
    tmp_stat = os.statvfs("/tmp")
    tmp_free_gib = (tmp_stat.f_bavail * tmp_stat.f_frsize) / (1024 ** 3)
    print(
        f"[vertex-vllm-entrypoint] Free space: /dev/shm={shm_free_gib:.1f} GiB, /tmp={tmp_free_gib:.1f} GiB",
        flush=True,
    )

    without_scheme = gcs_uri[len("gs://"):].rstrip("/")
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = (parts[1] + "/") if len(parts) > 1 else ""

    print(f"[vertex-vllm-entrypoint] Listing objects in gs://{bucket}/{prefix} ...", flush=True)
    token = get_gcp_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    all_items = []
    page_token = None
    while True:
        q = {"prefix": prefix, "maxResults": "1000"}
        if page_token:
            q["pageToken"] = page_token
        list_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?{urllib.parse.urlencode(q)}"
        req = urllib.request.Request(list_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for item in data.get("items", []):
            name = item["name"]
            if name.endswith("/"):
                continue
            rel_name = name[len(prefix):] if name.startswith(prefix) else os.path.basename(name)
            size = int(item.get("size", 0))
            all_items.append((name, rel_name, size))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    total_bytes = sum(x[2] for x in all_items)
    total_gib = total_bytes / (1024 ** 3)

    # Full local download if either /dev/shm or /tmp has >= 350 GiB free
    if shm_free_gib >= 350.0 or tmp_free_gib >= 350.0:
        local_dir = "/dev/shm/qwen3_model" if shm_free_gib >= 350.0 else "/tmp/qwen3_model"
        os.makedirs(local_dir, exist_ok=True)
        objects = [(bucket, name, rel_name, size, local_dir) for (name, rel_name, size) in all_items]
        objects.sort(key=lambda x: x[3])
        t_start = time.time()
        downloaded_bytes = 0
        completed_files = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(_download_one_object, obj) for obj in objects]
            for fut in concurrent.futures.as_completed(futures):
                rel_name, nbytes, dt = fut.result()
                downloaded_bytes += nbytes
                completed_files += 1
                if completed_files % 10 == 0 or completed_files == len(objects):
                    elapsed = max(0.001, time.time() - t_start)
                    gib_done = downloaded_bytes / (1024 ** 3)
                    print(
                        f"[vertex-vllm-entrypoint] Staged {completed_files}/{len(objects)} files "
                        f"({gib_done:.1f}/{total_gib:.1f} GiB) in {elapsed:.1f}s",
                        flush=True,
                    )
        return local_dir

    # Sliding-window RAM streaming mode (for 64 GiB /dev/shm)
    local_dir = "/tmp/qwen3_model"
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(SHARD_DIR, exist_ok=True)

    config_tasks = []
    shard_items = []
    for name, rel_name, size in all_items:
        base = os.path.basename(rel_name)
        if base.endswith(".safetensors") and base.startswith("model-"):
            shard_items.append((name, base, size))
            placeholder_path = os.path.join(local_dir, rel_name)
            os.makedirs(os.path.dirname(placeholder_path), exist_ok=True)
            if not os.path.exists(placeholder_path):
                with open(placeholder_path, "wb") as pf:
                    pf.write(b"\x00")
        else:
            config_tasks.append((bucket, name, rel_name, size, local_dir))

    print(
        f"[vertex-vllm-entrypoint] Downloading {len(config_tasks)} config/tokenizer files to {local_dir}...",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(_download_one_object, config_tasks))

    shard_items.sort(key=lambda x: x[1])
    manifest_path = "/tmp/qwen3_shard_manifest.json"
    with open(manifest_path, "w") as mf:
        json.dump(
            {
                "bucket": bucket,
                "tp_size": tp_size,
                "shards": shard_items,
            },
            mf,
        )

    with open(ACTIVE_FLAG, "w") as af:
        af.write("1")

    # Pre-stage the first 4 shards (~10.5 GiB) synchronously so worker startup hits ready shards immediately
    initial_batch = [
        (bucket, obj_name, f"{base}.ready", size, SHARD_DIR)
        for (obj_name, base, size) in shard_items[:4]
    ]
    print("[vertex-vllm-entrypoint] Pre-staging first 4 shards into /dev/shm/qwen3_shards...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(_download_one_object, initial_batch))

    print("[vertex-vllm-entrypoint] Launching background RAM shard prefetcher daemon...", flush=True)
    subprocess.Popen(
        [sys.executable, __file__, "--prefetch-daemon", manifest_path],
        close_fds=True,
    )
    return local_dir


def main_entrypoint():
    if len(sys.argv) >= 3 and sys.argv[1] == "--prefetch-daemon":
        run_prefetch_daemon(sys.argv[2])
        return

    os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "3600")
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
    os.environ.setdefault("PYTHONHASHSEED", "0")

    try:
        import fastapi
        import safetensors
        import vllm.model_executor.model_loader.weight_utils as wu
        test_app = fastapi.FastAPI()
        has_predict = "/predict" in {getattr(r, "path", None) for r in test_app.routes}
        print(
            f"[vertex-vllm-entrypoint] Verified hooks: safetensors={safetensors.safe_open.__name__}, "
            f"weight_utils={wu.safe_open.__name__}, fastapi_has_predict={has_predict}",
            flush=True,
        )
    except Exception as e:
        print(f"[vertex-vllm-entrypoint] Hook check warning: {e}", flush=True)

    try:
        df_out = subprocess.check_output(["df", "-h"], text=True)
        print(f"[vertex-vllm-entrypoint] Filesystem status:\n{df_out}", flush=True)
    except Exception:
        pass

    raw_args = sys.argv[1:]
    if raw_args and raw_args[0] == "serve":
        raw_args = raw_args[1:]

    model_path = None
    tp_size = 4
    filtered_args = []
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg in ("--model", "-m") and i + 1 < len(raw_args):
            model_path = raw_args[i + 1]
            i += 2
            continue
        elif arg.startswith("--model="):
            model_path = arg.split("=", 1)[1]
            i += 1
            continue
        elif arg in ("--tensor-parallel-size", "-tp") and i + 1 < len(raw_args):
            tp_size = int(raw_args[i + 1])
            filtered_args.extend([arg, raw_args[i + 1]])
            i += 2
            continue
        elif arg.startswith("--tensor-parallel-size="):
            tp_size = int(arg.split("=", 1)[1])
            filtered_args.append(arg)
            i += 1
            continue
        elif not arg.startswith("-") and model_path is None:
            model_path = arg
            i += 1
            continue
        filtered_args.append(arg)
        i += 1

    if model_path is None:
        model_path = os.environ.get(
            "AIP_STORAGE_URI", "gs://qwen3-8-flash-next-weights/vertex_model_ready"
        )

    local_model_path = stage_gcs_to_local(model_path, tp_size=tp_size)

    sys.argv = ["vllm", "serve", local_model_path] + filtered_args
    print(f"[vertex-vllm-entrypoint] Launching vLLM CLI with argv: {sys.argv}", flush=True)

    from vllm.entrypoints.cli.main import main
    sys.exit(main())


if __name__ == "__main__":
    main_entrypoint()
