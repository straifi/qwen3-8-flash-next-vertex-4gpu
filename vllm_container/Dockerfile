FROM vllm/vllm-openai:cu130-nightly

COPY sitecustomize.py /usr/local/lib/python3.12/dist-packages/sitecustomize.py
COPY vertex_entrypoint.py /usr/local/bin/vertex_entrypoint.py
RUN chmod +x /usr/local/bin/vertex_entrypoint.py && \
    rm -f /usr/lib/python3.12/sitecustomize.py /etc/python3.12/sitecustomize.py && \
    cp /usr/local/lib/python3.12/dist-packages/sitecustomize.py /usr/lib/python3.12/sitecustomize.py && \
    mkdir -p /etc/python3.12 && \
    cp /usr/local/lib/python3.12/dist-packages/sitecustomize.py /etc/python3.12/sitecustomize.py && \
    python3 -c "import os, site, shutil; [shutil.copy('/usr/local/lib/python3.12/dist-packages/sitecustomize.py', p + '/sitecustomize.py') for p in site.getsitepackages() if os.path.isdir(p) and p != '/usr/local/lib/python3.12/dist-packages']" && \
    python3 -m py_compile /usr/lib/python3.12/sitecustomize.py /usr/local/lib/python3.12/dist-packages/sitecustomize.py /usr/local/bin/vertex_entrypoint.py && \
    python3 -c "import safetensors, safetensors.torch, vllm.model_executor.model_loader.weight_utils as wu; print('vLLM hooks:', safetensors.safe_open.__name__, safetensors.torch.safe_open.__name__, wu.safe_open.__name__); assert safetensors.safe_open.__name__ == '_streaming_safe_open' and wu.safe_open.__name__ == '_streaming_safe_open'" && \
    python3 -c "import asyncio, fastapi, httpx; app = fastapi.FastAPI(); app.add_api_route('/v1/chat/completions', lambda: {'choices': [{'message': {'content': 'ok', 'reasoning_content': 'think'}}], 'usage': {'completion_tokens': 2}}, methods=['POST']); paths = [getattr(r, 'path', None) for r in app.routes]; print('FastAPI routes:', paths); assert '/predict' in paths; r1 = asyncio.run(httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost').post('/predict', json={'instances': [{'prompt': 'hi'}]})); assert r1.status_code == 200 and r1.json()['predictions'][0]['content'] == 'ok', r1.text; r2 = asyncio.run(httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost').post('/predict', json={'messages': [{'role': 'user', 'content': 'hi'}]})); assert r2.status_code == 200 and r2.json()['choices'][0]['message']['content'] == 'ok', r2.text; print('Verified /predict route for both :predict and :rawPredict!')"

ENV VLLM_ENGINE_READY_TIMEOUT_S=3600
ENV PYTHONHASHSEED=0
ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages

ENTRYPOINT ["python3", "/usr/local/bin/vertex_entrypoint.py"]
