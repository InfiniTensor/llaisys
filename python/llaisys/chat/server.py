import argparse
import json
import threading
import time
import uuid
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .service import ChatService


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    seed: int = 0
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)


def _response_model_name(service: ChatService, request_model: Optional[str]) -> str:
    return request_model or service.model_name


def _resolve_max_tokens(request: ChatCompletionRequest) -> int:
    if request.max_completion_tokens is not None:
        return int(request.max_completion_tokens)
    if request.max_tokens is not None:
        return int(request.max_tokens)
    return 128


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _sse_payload(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(model_path: str, device_name: str = "cpu") -> FastAPI:
    app = FastAPI(title="LLAISYS Chat Server", version="0.1.0")
    service = ChatService(model_path=model_path, device_name=device_name)
    serve_lock = threading.Lock()
    # 当前课程要求的是单用户服务，因此这里把模型实例与全局锁都挂到 app 上。
    app.state.chat_service = service
    app.state.serve_lock = serve_lock

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_path": service.model_path,
            "model_name": service.model_name,
            "device": service.device_name,
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must be non-empty")

        messages = [
            message.model_dump() if hasattr(message, "model_dump") else message.dict()
            for message in request.messages
        ]
        created = int(time.time())
        model_name = _response_model_name(service, request.model)
        max_tokens = _resolve_max_tokens(request)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        if not request.stream:
            # 非流式请求直接持锁跑完整次生成，避免不同请求共享同一个模型状态。
            with serve_lock:
                result = service.generate_chat(
                    messages=messages,
                    max_new_tokens=max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    seed=request.seed,
                )

            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.completion_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(
                    len(result.prompt_token_ids),
                    len(result.completion_token_ids),
                ),
            }

        # 流式场景需要把锁持有到整个生成过程结束，保证单用户串行。
        def event_stream():
            serve_lock.acquire()
            prompt_token_ids = []
            completion_token_ids = []
            try:
                _, prompt_token_ids = service.prepare_inputs(messages)
                # 先发一个只带 role 的首包，兼容 OpenAI 风格流式消费端。
                yield _sse_payload(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )

                for _, delta_text, completion_token_ids, _ in service.stream_completion(
                    prompt_token_ids,
                    max_new_tokens=max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    seed=request.seed,
                ):
                    if not delta_text:
                        continue
                    yield _sse_payload(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )

                yield _sse_payload(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": _usage(
                            len(prompt_token_ids),
                            len(completion_token_ids),
                        ),
                    }
                )
                # SSE 以 [DONE] 收尾，告诉客户端本轮生成已经结束。
                yield "data: [DONE]\n\n"
            finally:
                serve_lock.release()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="本地模型目录")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    app = create_app(model_path=args.model, device_name=args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
