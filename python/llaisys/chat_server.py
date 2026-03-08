import argparse
import json
import threading
import time
import uuid
from typing import List, Literal, Optional

import llaisys
from llaisys.models import Qwen2

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "FastAPI dependencies are missing. Install with: pip install fastapi uvicorn"
    ) from exc

try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "transformers is required for chat server. Install with: pip install transformers"
    ) from exc


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llaisys-qwen2"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    stream: bool = False


class ChatService:
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2(
            model_path,
            llaisys.DeviceType.NVIDIA if device == "nvidia" else llaisys.DeviceType.CPU,
        )
        self._lock = threading.Lock()

    def _build_input_ids(self, messages: List[ChatMessage]) -> List[int]:
        conversation = [{"role": m.role, "content": m.content} for m in messages]
        prompt = self.tokenizer.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.tokenizer.encode(prompt)

    def generate(self, req: ChatCompletionRequest):
        with self._lock:
            input_ids = self._build_input_ids(req.messages)
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=req.max_tokens,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature,
            )
            new_ids = output_ids[len(input_ids):]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            return text


def create_app(model_path: str, device: str = "cpu") -> FastAPI:
    app = FastAPI(title="LLAISYS Chat API", version="0.1.0")
    svc = ChatService(model_path, device)

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": "llaisys-qwen2", "object": "model", "owned_by": "llaisys"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if not req.stream:
            text = svc.generate(req)
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": text},
                        }
                    ],
                }
            )

        def event_stream():
            text = svc.generate(req)
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def main():
    parser = argparse.ArgumentParser(description="Run LLAISYS chat completion server")
    parser.add_argument("--model", required=True, type=str, help="Local model directory")
    parser.add_argument("--device", choices=["cpu", "nvidia"], default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    import uvicorn

    app = create_app(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
