import argparse
import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(
        "FastAPI is required for chat server. Install with: pip install fastapi uvicorn"
    ) from exc

from .engine import ChatEngine


def _dump_model(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    seed: Optional[int] = None
    stream: bool = False


def create_app(
    model_path: str,
    device: str = "cpu",
    max_new_tokens: int = 256,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 0.8,
) -> FastAPI:
    app = FastAPI(title="LLAISYS Chat Server", version="0.1.0")
    engine = ChatEngine(
        model_path=model_path,
        device=device,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    request_lock = threading.Lock()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        request_model = req.model
        request_messages = [_dump_model(m) for m in req.messages]
        request_max_new_tokens = req.max_new_tokens
        if request_max_new_tokens is None:
            request_max_new_tokens = req.max_tokens

        with request_lock:
            result = engine.complete(
                messages=request_messages,
                max_new_tokens=request_max_new_tokens,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature,
                seed=req.seed,
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if req.stream:
            generated_ids = result["generated_ids"]

            def event_stream():
                role_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

                for delta in engine.iter_text_deltas(generated_ids):
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                stop_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["assistant_text"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            },
        }
        return JSONResponse(payload)

    return app


def main():
    parser = argparse.ArgumentParser(description="LLAISYS OpenAI-compatible chat server")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument("--top-p", default=0.9, type=float)
    parser.add_argument("--temperature", default=0.8, type=float)
    args = parser.parse_args()

    app = create_app(
        model_path=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("uvicorn is required. Install with: pip install uvicorn") from exc

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
