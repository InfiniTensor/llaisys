## CUDA Backend for Chat Server

首先设置环境变量 `LLAISYS_DEVICE=nvidia` 来启用 CUDA backend.

## 流式输出

然后运行 `python/chat_server.py` 启用 OpenAI 风格的 API

然后可以用 `curl` 来测试流式输出

```bash
curl -N http://127.0.0.1:9108/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Accept: text/event-stream" \
-d '{"model":"qwen2","messages":[{"role":"user","content":"Hi who are you?"}],"stream":true,"max_tokens":64,"temperature":0.8,"top_p":0.9,"top_k":40}'
```

### TUI Chatting

```bash
python -m llaisys.chat.tui --url http://127.0.0.1:9108

# 或者用 uv
uv run python -m llaisys.chat.tui --url http://127.0.0.1:9108
```
