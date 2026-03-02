import argparse
import json
import threading
import time
import uuid
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependencies for chat server. Install with:\n"
        "  pip install fastapi uvicorn"
    ) from exc

from transformers import AutoTokenizer

# Prefer local python package source under repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import llaisys
from test_utils import llaisys_device

UI_HTML_PATH = Path(__file__).with_name("chat_web.html")


def parse_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def normalize_messages(raw_messages: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_messages, list) or len(raw_messages) == 0:
        raise ValueError("`messages` must be a non-empty list")

    out: List[Dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = item.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {role}")
        content = parse_message_content(item.get("content"))
        out.append({"role": role, "content": content})
    return out


class ChatEngine:
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device_name = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = llaisys.models.Qwen2(model_path, llaisys_device(device))
        self._infer_lock = threading.Lock()

    def _build_inputs(self, messages: List[Dict[str, str]]) -> List[int]:
        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.tokenizer.encode(prompt)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> Dict[str, Any]:
        with self._infer_lock:
            input_ids = self._build_inputs(messages)
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        completion_ids = out_ids[len(input_ids):]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return {
            "text": completion_text,
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(completion_ids),
        }

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> Iterable[Dict[str, Any]]:
        with self._infer_lock:
            input_ids = self._build_inputs(messages)
            if not hasattr(self.model, "generate_stream"):
                out_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                )
                completion_ids = out_ids[len(input_ids):]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                if completion_text:
                    yield {
                        "delta": completion_text,
                        "prompt_tokens": len(input_ids),
                        "completion_tokens": len(completion_ids),
                    }
                yield {
                    "delta": "",
                    "prompt_tokens": len(input_ids),
                    "completion_tokens": len(completion_ids),
                    "final_text": completion_text,
                }
                return

            generated_ids: List[int] = []
            previous_text = ""

            for token_id in self.model.generate_stream(
                input_ids,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            ):
                generated_ids.append(int(token_id))
                current_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                if current_text.startswith(previous_text):
                    delta = current_text[len(previous_text):]
                else:
                    # Fallback for rare decode normalization mismatch.
                    delta = self.tokenizer.decode([int(token_id)], skip_special_tokens=True)
                previous_text = current_text
                if delta:
                    yield {
                        "delta": delta,
                        "prompt_tokens": len(input_ids),
                        "completion_tokens": len(generated_ids),
                    }

            yield {
                "delta": "",
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(generated_ids),
                "final_text": previous_text,
            }


def create_app(engine: ChatEngine, served_model_name: str) -> FastAPI:
    app = FastAPI(title="LLAISYS Chat Server", version="0.1.0")

    @app.get("/")
    def chat_web() -> Any:
        if not UI_HTML_PATH.exists():
            raise HTTPException(status_code=404, detail="chat_web.html not found")
        return FileResponse(UI_HTML_PATH)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(payload: Dict[str, Any]) -> Any:
        try:
            messages = normalize_messages(payload.get("messages"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        stream = bool(payload.get("stream", False))
        top_k = int(payload.get("top_k", 1))
        top_p = float(payload.get("top_p", 1.0))
        temperature = float(payload.get("temperature", 1.0))
        max_new_tokens = int(payload.get("max_tokens", payload.get("max_new_tokens", 128)))
        max_new_tokens = max(1, max_new_tokens)

        request_model_name = payload.get("model")
        model_name = request_model_name if isinstance(request_model_name, str) else served_model_name

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not stream:
            result = engine.generate(
                messages=messages,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            response_obj = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                },
            }
            return JSONResponse(response_obj)

        def stream_iter():
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

            final_usage = None
            for item in engine.stream_generate(
                messages=messages,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            ):
                if "final_text" in item:
                    final_usage = item
                    break
                delta_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": item["delta"]},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(delta_chunk, ensure_ascii=False)}\n\n"

            finish_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"

            if final_usage is not None:
                usage_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "usage": {
                        "prompt_tokens": final_usage["prompt_tokens"],
                        "completion_tokens": final_usage["completion_tokens"],
                        "total_tokens": (
                            final_usage["prompt_tokens"] + final_usage["completion_tokens"]
                        ),
                    },
                    "choices": [],
                }
                yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_iter(), media_type="text/event-stream")

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="LLAISYS OpenAI-style Chat Server")
    parser.add_argument("--model", required=True, type=str, help="Path to model directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--served-model-name", default="llaisys-qwen2", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    engine = ChatEngine(model_path=args.model, device=args.device)
    app = create_app(engine, served_model_name=args.served_model_name)

    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing uvicorn. Install with:\n"
            "  pip install uvicorn"
        ) from exc

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
