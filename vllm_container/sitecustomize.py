"""
Automatic safetensors RAM-streamer hook AND Vertex AI `/predict` FastAPI route injector
for vLLM multi-process workers (TP=4 on 4x NVIDIA RTX PRO 6000).
Installed into /usr/lib/python3.12/sitecustomize.py, /etc/python3.12/sitecustomize.py,
and /usr/local/lib/python3.12/dist-packages/sitecustomize.py so that:
1. The main process, EngineCore, and all 4 Tensor-Parallel worker processes transparently
   read .safetensors shards from /dev/shm/qwen3_shards/ as they are prefetched from GCS,
   and release each shard from /dev/shm as soon as all TP workers finish loading it.
2. Every FastAPI application instance created by vLLM automatically registers a POST /predict
   route supporting both Vertex AI standard `:predict` ({"instances": [...]}) and
   `:rawPredict` (OpenAI /v1/chat/completions payload).
"""
import contextlib
import os
import time

SHARD_DIR = "/dev/shm/qwen3_shards"
ACTIVE_FLAG = "/dev/shm/qwen3_stream_active"
_LOGGED_FIRST_OPEN = False

# 1. Install safetensors sliding-window /dev/shm streamer
try:
    import safetensors
    import safetensors.torch

    _orig_safe_open = safetensors.safe_open
    _orig_load_file = safetensors.torch.load_file

    @contextlib.contextmanager
    def _streaming_safe_open(filename, *args, **kwargs):
        global _LOGGED_FIRST_OPEN
        fname_str = str(filename)
        basename = os.path.basename(fname_str)
        if (
            basename.endswith(".safetensors")
            and basename.startswith("model-")
            and os.path.exists(ACTIVE_FLAG)
        ):
            ready_path = os.path.join(SHARD_DIR, f"{basename}.ready")
            req_path = os.path.join(SHARD_DIR, f"{basename}.req")
            in_use_dir = os.path.join(SHARD_DIR, f"{basename}.in_use")
            done_dir = os.path.join(SHARD_DIR, f"{basename}.done")
            os.makedirs(in_use_dir, exist_ok=True)
            os.makedirs(done_dir, exist_ok=True)

            if not _LOGGED_FIRST_OPEN:
                _LOGGED_FIRST_OPEN = True
                print(
                    f"[sitecustomize pid={os.getpid()}] Intercepted safe_open({basename}) -> streaming from {SHARD_DIR}",
                    flush=True,
                )

            lease_file = os.path.join(in_use_dir, f"{os.getpid()}_{time.monotonic_ns()}")
            try:
                with open(lease_file, "w") as lf:
                    lf.write("1")
            except Exception:
                pass

            t_wait = time.time()
            opened_handle = None
            try:
                while opened_handle is None:
                    if not os.path.exists(ready_path):
                        try:
                            with open(req_path, "w") as rf:
                                rf.write(str(os.getpid()))
                        except Exception:
                            pass
                        time.sleep(0.05)
                        if time.time() - t_wait > 1800:
                            raise TimeoutError(f"Timed out waiting for shard {basename} in {SHARD_DIR}")
                        continue
                    try:
                        opened_handle = _orig_safe_open(ready_path, *args, **kwargs)
                    except FileNotFoundError:
                        opened_handle = None
                        time.sleep(0.05)

                with opened_handle as f:
                    yield f
            finally:
                try:
                    with open(os.path.join(done_dir, str(os.getpid())), "w") as df:
                        df.write("1")
                except Exception:
                    pass
                try:
                    if os.path.exists(lease_file):
                        os.remove(lease_file)
                except Exception:
                    pass
        else:
            with _orig_safe_open(filename, *args, **kwargs) as f:
                yield f

    def _streaming_load_file(filename, device="cpu"):
        result = {}
        with _streaming_safe_open(filename, framework="pt", device=device) as f:
            for k in f.keys():
                result[k] = f.get_tensor(k)
        return result

    safetensors.safe_open = _streaming_safe_open
    safetensors.torch.safe_open = _streaming_safe_open
    safetensors.torch.load_file = _streaming_load_file
except Exception as e:
    print(f"[sitecustomize] Warning: could not install safetensors streamer: {e}", flush=True)


# 2. Install FastAPI `/predict` route hook so Vertex AI `:predict` and `:rawPredict` always succeed
try:
    import fastapi
    from fastapi import Request
    from fastapi.responses import JSONResponse

    _orig_fastapi_init = fastapi.FastAPI.__init__

    def _attach_vertex_predict_route(app: "fastapi.FastAPI"):
        existing_paths = {getattr(r, "path", None) for r in getattr(app, "routes", [])}
        if "/predict" in existing_paths:
            return

        @app.post("/predict")
        async def vertex_predict(request: Request):
            body = await request.json()
            from httpx import AsyncClient, ASGITransport

            target_app = request.app or app
            async with AsyncClient(
                transport=ASGITransport(app=target_app), base_url="http://localhost"
            ) as client:
                # Direct OpenAI chat completion payload via :rawPredict
                if isinstance(body, dict) and "messages" in body and "instances" not in body:
                    resp = await client.post("/v1/chat/completions", json=body, timeout=300.0)
                    return JSONResponse(resp.json(), status_code=resp.status_code)

                # Direct OpenAI text completion payload via :rawPredict
                if isinstance(body, dict) and "prompt" in body and "instances" not in body:
                    messages = [{"role": "user", "content": body["prompt"]}]
                    chat_payload = {
                        k: v for k, v in body.items() if k != "prompt"
                    }
                    chat_payload["messages"] = messages
                    chat_payload.setdefault("max_tokens", 512)
                    resp = await client.post("/v1/chat/completions", json=chat_payload, timeout=300.0)
                    return JSONResponse(resp.json(), status_code=resp.status_code)

                # Standard Vertex AI :predict format: {"instances": [...], "parameters": {...}}
                instances = body.get("instances", [body]) if isinstance(body, dict) else [body]
                params = body.get("parameters", {}) if isinstance(body, dict) else {}
                predictions = []

                for inst in instances:
                    if isinstance(inst, str):
                        messages = [{"role": "user", "content": inst}]
                        req_params = dict(params)
                    elif isinstance(inst, dict):
                        if "messages" in inst:
                            messages = inst["messages"]
                        elif "prompt" in inst:
                            messages = [{"role": "user", "content": inst["prompt"]}]
                        else:
                            messages = [{"role": "user", "content": str(inst)}]
                        req_params = {
                            **params,
                            **{k: v for k, v in inst.items() if k not in ("messages", "prompt")},
                        }
                    else:
                        messages = [{"role": "user", "content": str(inst)}]
                        req_params = dict(params)

                    chat_payload = {
                        "messages": messages,
                        "max_tokens": req_params.get("max_tokens", 512),
                        "temperature": req_params.get("temperature", 0.7),
                    }
                    for opt_key in ("top_p", "top_k", "stop", "chat_template_kwargs", "model"):
                        if opt_key in req_params:
                            chat_payload[opt_key] = req_params[opt_key]

                    resp = await client.post("/v1/chat/completions", json=chat_payload, timeout=300.0)
                    if resp.status_code != 200:
                        return JSONResponse(
                            {"error": resp.text, "status_code": resp.status_code},
                            status_code=resp.status_code,
                        )
                    data = resp.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        msg = data["choices"][0].get("message", {})
                        predictions.append(
                            {
                                "content": msg.get("content", ""),
                                "reasoning_content": msg.get("reasoning_content"),
                                "usage": data.get("usage", {}),
                            }
                        )
                    else:
                        predictions.append(data)

                return JSONResponse({"predictions": predictions})

    def _patched_fastapi_init(self, *args, **kwargs):
        _orig_fastapi_init(self, *args, **kwargs)
        try:
            _attach_vertex_predict_route(self)
        except Exception as e:
            print(f"[sitecustomize] Warning attaching /predict route: {e}", flush=True)

    fastapi.FastAPI.__init__ = _patched_fastapi_init
except Exception as e:
    print(f"[sitecustomize] Warning: could not install FastAPI /predict hook: {e}", flush=True)
