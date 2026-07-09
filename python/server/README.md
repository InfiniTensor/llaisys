# LLAISYS Chat Server (Project #3)

A minimal Python chat server built on top of `llaisys.models.Qwen2`.

## Install deps

```bash
python3 -m pip install fastapi uvicorn
```

## Start server

```bash
cd python
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open chat UI:

```bash
http://127.0.0.1:8000/
```

Optional environment variables:

- `LLAISYS_MODEL_PATH`: local model path (default: `/home/vankari/code/DeepSeek-R1-Distill-Qwen-1.5B/`; if path does not exist, falls back to HF model)
- `LLAISYS_DEVICE`: `cpu` or `nvidia` (default: `cpu`)
- `LLAISYS_MAX_NEW_TOKENS` (default: `128`)
- `LLAISYS_TOP_K` (default: `50`)
- `LLAISYS_TOP_P` (default: `0.8`)
- `LLAISYS_TEMPERATURE` (default: `0.8`)
- `LLAISYS_USE_CACHE` (default: `true`)
- `LLAISYS_SERVE_MODEL_NAME` (default: `llaisys-qwen2`)

## API

### Health

```bash
curl http://127.0.0.1:8000/healthz
```

### Chat completion

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llaisys-qwen2",
    "messages": [{"role": "user", "content": "你好，介绍一下你自己"}],
    "max_tokens": 64,
    "top_k": 50,
    "top_p": 0.8,
    "temperature": 0.8,
    "stream": false
  }'
```

### Streaming completion (SSE)

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "讲个笑话"}],
    "stream": true
  }'
```
