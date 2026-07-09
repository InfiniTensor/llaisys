import json
import re
import time
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .config import CONFIG
from .engine import ENGINE
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatResponseMessage,
    Usage,
)

app = FastAPI(title="LLAISYS Chat Server", version="0.1.0")
UI_INDEX = Path(__file__).resolve().parent / "ui" / "index.html"


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = None


@app.get("/")
def chat_ui():
    if not UI_INDEX.exists():
        raise HTTPException(status_code=404, detail="UI page not found")
    return FileResponse(UI_INDEX)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": CONFIG.device, "model": CONFIG.default_model_name}


def _chunk_payload(chunk_id: str, model_name: str, content: str, finish_reason=None):
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }

def _normalize_think_text(text: str, prompt_has_think_tag: bool) -> str:
    if not text:
        return text
    has_open = re.search(r"<think>", text) is not None
    has_close = re.search(r"</think>", text) is not None
    if prompt_has_think_tag and has_close and not has_open:
        text = f"<think>{text}"
        has_open = True
    if prompt_has_think_tag and has_open and not has_close:
        text = f"{text}</think>"
    return text


def _prompt_has_think_tag(messages: List[Dict[str, str]]) -> bool:
    prompt = ENGINE._tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    return "<think>" in prompt or "</think>" in prompt


def _filter_newlines(text: str) -> str:
    if not text:
        return text
    return text.replace("\n", "")


def _longest_suffix_prefix(value: str, token: str) -> int:
    upper = min(len(value), len(token) - 1)
    for n in range(upper, 0, -1):
        if value.endswith(token[:n]):
            return n
    return 0


def _normalize_stream_piece(piece: str, prompt_has_think_tag: bool, seen_open_tag: bool):
    start_tag = "<think>"
    end_tag = "</think>"
    source = piece
    out_parts = []

    while source:
        start_idx = source.find(start_tag)
        end_idx = source.find(end_tag)

        next_idx = -1
        tag_type = ""
        if start_idx != -1 and (end_idx == -1 or start_idx < end_idx):
            next_idx = start_idx
            tag_type = "start"
        elif end_idx != -1:
            next_idx = end_idx
            tag_type = "end"

        if next_idx == -1:
            keep = max(
                _longest_suffix_prefix(source, start_tag),
                _longest_suffix_prefix(source, end_tag),
            )
            if keep > 0:
                out_parts.append(source[:-keep])
                return "".join(out_parts), source[-keep:], seen_open_tag
            out_parts.append(source)
            return "".join(out_parts), "", seen_open_tag

        out_parts.append(source[:next_idx])
        source = source[next_idx + (len(start_tag) if tag_type == "start" else len(end_tag)) :]

        if tag_type == "start":
            if not seen_open_tag:
                out_parts.append(start_tag)
            seen_open_tag = True
        else:
            if prompt_has_think_tag and not seen_open_tag:
                out_parts.append(start_tag)
                seen_open_tag = True
            out_parts.append(end_tag)

    return "".join(out_parts), "", seen_open_tag


def _stream_completion(req: ChatCompletionRequest, model_name: str) -> Iterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    messages = [m.model_dump() for m in req.messages]
    prompt_has_think_tag = _prompt_has_think_tag(messages)
    seen_open_tag = False
    carry = ""

    def flush_payload(content: str) -> Iterator[str]:
        content = _filter_newlines(content)
        if not content:
            return
        payload = _chunk_payload(chunk_id, model_name, content)
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    if prompt_has_think_tag:
        seen_open_tag = True
        yield from flush_payload("<think>")

    for piece in ENGINE.stream_generate(
        messages=messages,
        max_tokens=req.max_tokens or CONFIG.default_max_new_tokens,
        top_k=req.top_k if req.top_k is not None else CONFIG.default_top_k,
        top_p=req.top_p if req.top_p is not None else CONFIG.default_top_p,
        temperature=req.temperature if req.temperature is not None else CONFIG.default_temperature,
        use_cache=req.use_cache if req.use_cache is not None else CONFIG.default_use_cache,
        session_id=req.session_id,
    ):
        if not piece:
            continue
        merged = carry + piece
        normalized_piece, carry, seen_open_tag = _normalize_stream_piece(
            merged,
            prompt_has_think_tag,
            seen_open_tag,
        )
        if normalized_piece:
            yield from flush_payload(normalized_piece)

    if carry:
        final_piece, _, _ = _normalize_stream_piece(carry, prompt_has_think_tag, seen_open_tag)
        if final_piece:
            yield from flush_payload(final_piece)

    done_payload = _chunk_payload(chunk_id, model_name, "", finish_reason="stop")
    yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    messages = [m.model_dump() for m in req.messages]
    model_name = req.model or CONFIG.default_model_name
    if req.stream:
        return StreamingResponse(_stream_completion(req, model_name), media_type="text/event-stream")

    result = ENGINE.generate(
        messages=messages,
        max_tokens=req.max_tokens or CONFIG.default_max_new_tokens,
        top_k=req.top_k if req.top_k is not None else CONFIG.default_top_k,
        top_p=req.top_p if req.top_p is not None else CONFIG.default_top_p,
        temperature=req.temperature if req.temperature is not None else CONFIG.default_temperature,
        use_cache=req.use_cache if req.use_cache is not None else CONFIG.default_use_cache,
        session_id=req.session_id,
    )

    completion_text = _normalize_think_text(result["completion_text"], _prompt_has_think_tag(messages)).strip()
    completion_text = _filter_newlines(completion_text)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    response = ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=result["created"],
        model=model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatResponseMessage(role="assistant", content=completion_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=len(result["prompt_ids"]),
            completion_tokens=len(result["completion_ids"]),
            total_tokens=len(result["prompt_ids"]) + len(result["completion_ids"]),
        ),
    )
    return response


@app.post("/v1/sessions")
def create_session(req: SessionCreateRequest):
    session_id = ENGINE.create_session(req.session_id)
    return {"session_id": session_id}


@app.get("/v1/sessions")
def list_sessions():
    return {"sessions": ENGINE.list_sessions()}


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str):
    ENGINE.delete_session(session_id)
    return {"deleted": session_id}


@app.get("/v1/sessions/{session_id}")
def get_session(session_id: str):
    try:
        messages = ENGINE.get_session_messages(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "messages": messages}
