import os
import time
import uuid
import json
import threading
from pathlib import Path
from typing import List, Optional, Iterable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse, JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
from transformers import AutoTokenizer

import llaisys

# define Message format
class ChatMessage(BaseModel):
    role: str
    content: str

# define Request format
class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 50
    stream: bool = False
    seed: int = 0


class AppState:
    tokenizer = None
    model = None
    model_path = None
    model_name = None
    device = None


state = AppState()
app = FastAPI()
model_lock = threading.Lock()

WEBUI_INDEX = Path(__file__).resolve().parent.parent / "webUI" / "index.html"
WEBUI_DIR = WEBUI_INDEX.parent.parent
WEBUI_ENABLED = WEBUI_INDEX.is_file()
if WEBUI_ENABLED:
    app.mount("/webUI", StaticFiles(directory=WEBUI_DIR), name="webUI")

# map device name to llaisys device type
def llaisys_device(device_name: str):
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    if device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    raise ValueError(f"Unsupported device name: {device_name}")

# set default values for model path, device, and model name 
# if not provided
def ensure_state_loaded():
    if state.model is not None and state.tokenizer is not None:
        return
    model_path = os.environ.get("LLAISYS_MODEL_PATH", "./models")
    if not model_path:
        raise RuntimeError("LLAISYS_MODEL_PATH is required")
    device = os.environ.get("LLAISYS_DEVICE", "cpu")
    model_name = os.environ.get("LLAISYS_MODEL_NAME", "llaisys-qwen2")
    state.model_path = model_path
    state.device = device
    state.model_name = model_name
    state.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    state.model = llaisys.models.Qwen2(model_path, llaisys_device(device))

# build prompt from messages
def build_prompt(messages: List[ChatMessage]) -> str:
    conversation = [{"role": m.role, "content": m.content} for m in messages]
    return state.tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

# decode output token id and mask prompt tokens
def decode_completion(input_ids: List[int], output_ids: List[int]) -> str:
    prompt_text = state.tokenizer.decode(input_ids, skip_special_tokens=True)
    full_text = state.tokenizer.decode(output_ids, skip_special_tokens=True)
    return full_text[len(prompt_text):]

# generate full completion
def generate_full(
    input_ids: List[int],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
):
    output_ids = state.model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        seed=seed,
    )
    completion_text = decode_completion(input_ids, output_ids)
    completion_tokens = max(0, len(output_ids) - len(input_ids))
    return completion_text, completion_tokens

# stream mode
def generate_stream(
    input_ids: List[int],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> Iterable[str]:
    tokens = list(input_ids)
    prompt_text = state.tokenizer.decode(tokens, skip_special_tokens=True)
    prev_text = prompt_text
    if max_tokens <= 0:
        return
    # generate first output token, then append it to prompt
    next_token = state.model._infer(tokens, temperature, top_k, top_p, seed)
    tokens.append(next_token)
    text = state.tokenizer.decode(tokens, skip_special_tokens=True)
    delta = text[len(prev_text):]
    prev_text = text
    # if we decode a non-empty delta, yield it, that's the first token
    if delta:
        yield delta
    # continue generate subsequent tokens, until max_tokens or end_token
    for _ in range(max_tokens - 1):
        if tokens[-1] == state.model._end_token:
            break
        seed += 1
        # because of kv-cache, we only need to infer the last token
        next_token = state.model._infer([tokens[-1]], temperature, top_k, top_p, seed)
        tokens.append(next_token)
        text = state.tokenizer.decode(tokens, skip_special_tokens=True)
        delta = text[len(prev_text):]
        prev_text = text
        if delta:
            yield delta


def sse_chunk(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    ensure_state_loaded()
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")
    prompt = build_prompt(req.messages)
    input_ids = state.tokenizer.encode(prompt)
    max_tokens = req.max_tokens if req.max_tokens is not None else 128
    created = int(time.time())
    request_id = "chatcmpl-" + uuid.uuid4().hex

    if not req.stream:
        with model_lock:
            completion_text, completion_tokens = generate_full(
                input_ids=input_ids,
                max_tokens=max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                seed=req.seed,
            )
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model or state.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": completion_tokens,
                "total_tokens": len(input_ids) + completion_tokens,
            },
        }
        return JSONResponse(response)

    def event_stream():
        model_lock.acquire()
        try:
            yield sse_chunk(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model or state.model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            )
            for delta in generate_stream(
                input_ids=input_ids,
                max_tokens=max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                seed=req.seed,
            ):
                yield sse_chunk(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model or state.model_name,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                )
            yield sse_chunk(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model or state.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield "data: [DONE]\n\n"
        finally:
            model_lock.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
def chat_ui():
    if WEBUI_ENABLED:
        return FileResponse(WEBUI_INDEX)
    raise HTTPException(status_code=404, detail="Web UI not found")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--device", default="nvidia")
    parser.add_argument("--model_name", default="llaisys-qwen2")
    args = parser.parse_args()

    if args.model_path:
        os.environ["LLAISYS_MODEL_PATH"] = args.model_path
    os.environ["LLAISYS_DEVICE"] = args.device
    os.environ["LLAISYS_MODEL_NAME"] = args.model_name

    uvicorn.run(app, host=args.host, port=args.port)
