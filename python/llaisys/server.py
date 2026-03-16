"""
LLAISYS Chat Server — OpenAI-compatible chat-completion API.

Usage:
    python -m llaisys.server --model /path/to/model [--host 0.0.0.0] [--port 8000]
"""

import argparse
import json
import time
import uuid
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from transformers import AutoTokenizer

from .models.qwen2 import Qwen2
from .libllaisys import DeviceType

app = FastAPI(title="LLAISYS Chat Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Optional[Qwen2] = None
_tokenizer = None
_lock = threading.Lock()
_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# ── Pydantic schemas (OpenAI-compatible) ──────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen2"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=512, alias="max_tokens")
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    stream: Optional[bool] = False

class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_prompt(messages: List[ChatMessage]) -> str:
    conversation = [{"role": m.role, "content": m.content} for m in messages]
    return _tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        tokenize=False,
    )


def _generate_stream_chunks(request_id, model_name, input_ids, temperature, top_k, top_p, max_tokens):
    """Yield SSE data chunks for streaming responses."""
    _model.reset_kvcache()

    for token_id in _model.generate_stream(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    ):
        text = _tokenizer.decode([token_id], skip_special_tokens=True)
        if not text:
            continue
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    done_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────

def _find_static_dir() -> Path:
    candidates = [
        Path(__file__).parent / "static",
        Path(__file__).resolve().parent / "static",
        Path(__file__).resolve().parent.parent.parent / "python" / "llaisys" / "static",
    ]
    for c in candidates:
        if (c / "index.html").is_file():
            return c
    return candidates[0]

_static_dir = _find_static_dir()


@app.get("/")
async def index():
    html_path = _static_dir / "index.html"
    if not html_path.is_file():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=500)
    return FileResponse(html_path)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": _model_name, "object": "model", "owned_by": "llaisys"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt_text = _build_prompt(req.messages)
    input_ids = _tokenizer.encode(prompt_text)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    temperature = req.temperature or 0.8
    top_k = req.top_k or 50
    top_p = req.top_p or 0.9
    max_tokens = req.max_tokens or 512

    if req.stream:
        def locked_stream():
            with _lock:
                yield from _generate_stream_chunks(
                    request_id, req.model, input_ids,
                    temperature, top_k, top_p, max_tokens,
                )
        return StreamingResponse(
            locked_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    with _lock:
        _model.reset_kvcache()
        output_tokens = _model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    new_tokens = output_tokens[len(input_ids):]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=req.model,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))],
        usage=ChatUsage(
            prompt_tokens=len(input_ids),
            completion_tokens=len(new_tokens),
            total_tokens=len(output_tokens),
        ),
    )


# ── Server bootstrap ──────────────────────────────────────────────────────

def init_model(model_path: str, device: str = "cpu"):
    global _model, _tokenizer, _model_name
    device_type = DeviceType.CPU if device == "cpu" else DeviceType.NVIDIA

    from huggingface_hub import snapshot_download
    local_path = snapshot_download(model_path)

    print(f"Loading tokenizer from {local_path} ...")
    _tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    print(f"Loading LLAISYS model on {device} from {local_path} ...")
    _model = Qwen2(local_path, device_type)
    print("Model loaded.")


def main():
    parser = argparse.ArgumentParser(description="LLAISYS Chat Server")
    parser.add_argument("--model", required=True, type=str, help="Path to model directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    init_model(args.model, args.device)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
