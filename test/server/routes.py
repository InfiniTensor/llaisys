"""
routes.py — Chat completion API endpoints + session management

This module implements:
  - /v1/chat/completions  — generate a response for a given session
  - /v1/sessions          — list, create, and delete chat sessions

Session management:
  Each session is an independent conversation with its own message history.
  Sessions are stored in memory (SESSIONS dict) and identified by a UUID.
  The client sends a session_id with every chat request so the server knows
  which history to use.

Key concepts:
  - Streaming vs non-streaming: controlled by the `stream` field in the request.
    Streaming uses Server-Sent Events (SSE), which lets the browser receive
    tokens one by one as they are generated, instead of waiting for the full
    response.
  - run_in_executor: The C++ model inference is synchronous (blocking). FastAPI
    runs on an async event loop (asyncio), so if we call blocking code directly
    it freezes the entire server. run_in_executor offloads the blocking call to
    a thread pool, keeping the event loop free to handle SSE writes.
  - Incremental decode: We decode the full list of generated tokens on every
    step and take the new suffix as the delta. This avoids BPE boundary issues
    where a single token only produces visible text when combined with the next.
  - <think> filtering: DeepSeek-R1 wraps its chain-of-thought reasoning in
    <think>...</think> tags. We strip these before sending text to the client.
"""

import asyncio
import json
import re
import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()

# Global model and tokenizer, set once at server startup via set_model().
MODEL = None
TOKENIZER = None

# ── Session store ─────────────────────────────────────────────────────────────
# Each session is a dict with:
#   id      — unique UUID string
#   title   — display name (set to the first user message, truncated)
#   history — list of {role, content} dicts, the full conversation so far
#
# In a production system this would be persisted to a database, but for a
# single-user educational server, an in-memory dict is sufficient.
SESSIONS: dict = {}  # session_id -> session dict


def _new_session() -> dict:
    """Create a new empty session and add it to the store."""
    sid = uuid.uuid4().hex[:8]
    session = {"id": sid, "title": "New chat", "history": []}
    SESSIONS[sid] = session
    return session


def set_model(model, tokenizer):
    """Called by main.py after loading the model to make it available here."""
    global MODEL, TOKENIZER
    MODEL = model
    TOKENIZER = tokenizer


# ── Request / Response schemas ────────────────────────────────────────────────
# Pydantic models validate incoming JSON automatically and produce clear error
# messages when fields are missing or out of range.

class ChatMessage(BaseModel):
    role: str     # "user" or "assistant"
    content: str  # the message text


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: str = ""  # Which session to use; empty = create a new one
    # Sampling parameters — see sample operator for algorithm details
    temperature: float = Field(default=0.8, ge=0.1, le=2.0,
        description="Higher = more random, lower = more deterministic")
    top_k: int = Field(default=50, ge=0,
        description="Keep only the top-K most likely tokens (0 = disabled)")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0,
        description="Nucleus sampling: keep tokens whose cumulative prob >= top_p")
    max_tokens: int = Field(default=512, gt=0,
        description="Maximum number of tokens to generate")
    stream: bool = False  # If True, use SSE streaming response


# ── Helper functions ──────────────────────────────────────────────────────────

def _clean_output(text: str) -> str:
    """
    Remove model-specific artifacts from generated text.

    DeepSeek-R1 outputs two kinds of noise we need to strip:
    1. <think>...</think> blocks — the model's internal chain-of-thought
       reasoning. We don't want to show this to the user.
    2. Special tokens like <｜end▁of▁sentence｜> — control tokens that the
       tokenizer emits when skip_special_tokens=False. We need to decode with
       skip_special_tokens=False so that <think> tags are preserved for
       filtering, but then we must manually remove the other special tokens.

    Note: the chat template appends "<think>\n" to the prompt, so generated
    tokens start *inside* the think block (no opening <think> tag). We handle
    both cases: full <think>...</think> blocks and orphaned </think> tags.
    """
    # Strip full chain-of-thought block when both tags are present
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # Strip everything up to and including an orphaned </think> tag
    # (happens when <think> was part of the prompt template, not the output)
    text = re.sub(r"^[\s\S]*?</think>", "", text)
    # Strip special tokens (both ASCII <|...|> and fullwidth <｜...｜> variants)
    text = re.sub(r"<[|｜][^|｜]*[|｜]>", "", text)
    return text.strip()


def _check_model():
    """Raise HTTP 503 if the model hasn't been loaded yet."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


def _encode(messages: List[ChatMessage]) -> List[int]:
    """
    Apply the model's chat template and encode to token IDs.

    apply_chat_template formats the conversation history into the exact
    prompt format the model was trained on (e.g. <|User|>...<|Assistant|>).
    This is important — using the wrong format degrades response quality.
    """
    chat = [{"role": m.role, "content": m.content} for m in messages]
    text = TOKENIZER.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return TOKENIZER.encode(text)


def _make_chunk(cid: str, content: str) -> str:
    """
    Format a text delta as an SSE data line in OpenAI chunk format.

    SSE (Server-Sent Events) protocol: each event is a line starting with
    "data: " followed by the payload, terminated by two newlines.
    The client reads these line by line and updates the UI incrementally.
    """
    data = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(data)}\n\n"


def _make_done_chunk(cid: str) -> str:
    """
    Send the final SSE event signaling end of stream.

    The OpenAI streaming protocol ends with a chunk where finish_reason="stop"
    followed by a special "data: [DONE]" line. The client uses [DONE] to know
    it can close the connection.
    """
    data = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"


# ── Session endpoints ─────────────────────────────────────────────────────────

@router.get("/v1/sessions")
async def list_sessions():
    """Return all sessions (id + title only, not full history)."""
    return [{"id": s["id"], "title": s["title"]} for s in SESSIONS.values()]


@router.post("/v1/sessions")
async def create_session():
    """Create a new empty session and return it."""
    session = _new_session()
    return {"id": session["id"], "title": session["title"]}


@router.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Return a session's full history (id, title, history)."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    s = SESSIONS[session_id]
    return {"id": s["id"], "title": s["title"], "history": s["history"]}


@router.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session by ID."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    del SESSIONS[session_id]
    return {"deleted": session_id}


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Main chat endpoint, compatible with the OpenAI chat completion API.

    The client sends the full message list and a session_id. The server
    updates the session's history after each successful generation so that
    subsequent requests in the same session have full context.
    """
    _check_model()

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    # Resolve or create the session
    if req.session_id and req.session_id in SESSIONS:
        session = SESSIONS[req.session_id]
    else:
        # No valid session_id provided — create a new session automatically
        session = _new_session()

    # The client sends the full history it knows about; we use that as the
    # source of truth (the client may have added a new user message).
    session["history"] = [{"role": m.role, "content": m.content} for m in req.messages]

    # Set the session title from the first user message (truncated to 30 chars)
    user_msgs = [m for m in req.messages if m.role == "user"]
    if user_msgs and session["title"] == "New chat":
        session["title"] = user_msgs[0].content[:30]

    # Encode the full conversation history into a flat token sequence.
    # The model sees the entire history on every turn — this is how it
    # maintains context. The KV cache in the C++ backend avoids recomputing
    # attention for tokens it has already seen.
    tokens = _encode(req.messages)
    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if req.stream:
        return StreamingResponse(
            _stream_generate(tokens, cid, req, session),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                # Send session_id in header so client can update its state
                "X-Session-Id": session["id"],
            },
        )

    # ── Non-streaming path ────────────────────────────────────────────────────
    generated = []
    current_tokens = list(tokens)
    for _ in range(req.max_tokens):
        next_token = MODEL.generate(
            current_tokens, max_new_tokens=1,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        )[-1]
        current_tokens.append(next_token)
        generated.append(next_token)
        if MODEL._end_token >= 0 and next_token == MODEL._end_token:
            break

    text = _clean_output(TOKENIZER.decode(generated, skip_special_tokens=False))

    # Persist the assistant reply into the session history
    session["history"].append({"role": "assistant", "content": text})

    return {
        "id": cid,
        "session_id": session["id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(tokens),
            "completion_tokens": len(generated),
            "total_tokens": len(tokens) + len(generated),
        },
    }


async def _stream_generate(tokens: List[int], cid: str, req: ChatCompletionRequest, session: dict):
    """
    Async generator that yields SSE chunks one token at a time.

    The challenge: model inference is synchronous C++ code, but we need to
    yield SSE chunks asynchronously. Solution: run_in_executor offloads each
    blocking inference call to a thread pool worker, then awaits the result.
    This lets the asyncio event loop stay responsive and flush each chunk to
    the HTTP client immediately after it's generated.

    Incremental decode strategy:
      We decode the full list of generated tokens on every step and compare
      with the previous decoded length to get the new text delta. This is
      necessary because some tokens only produce visible characters when
      decoded together (BPE subword tokenization). Decoding token-by-token
      would produce garbled output for multi-byte characters.

    History persistence strategy:
      Add assistant message to history immediately with empty content, then
      update it after each token. This ensures partial responses are saved
      even if the client disconnects mid-stream.
    """
    loop = asyncio.get_event_loop()
    current_tokens = list(tokens)
    generated = []
    prev_clean_len = 0  # Length of clean text already sent to the client

    # Add assistant message placeholder to history immediately
    # This ensures we save partial responses if client disconnects
    assistant_msg = {"role": "assistant", "content": ""}
    session["history"].append(assistant_msg)

    def _infer_one():
        """Synchronous inference call — runs in a thread pool worker."""
        return MODEL.generate(
            current_tokens, max_new_tokens=1,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        )[-1]

    try:
        for _ in range(req.max_tokens):
            # Offload blocking C++ call to thread pool, await the result.
            # While waiting, the event loop can handle other tasks (e.g. flush
            # already-yielded chunks to the HTTP client).
            next_token = await loop.run_in_executor(None, _infer_one)
            current_tokens.append(next_token)
            generated.append(next_token)

            # Stop generation when end-of-sequence token is produced
            if MODEL._end_token >= 0 and next_token == MODEL._end_token:
                break

            # Decode all generated tokens so far (not just the latest one).
            # This handles BPE boundaries correctly: a token like "▁Hello"
            # only decodes to " Hello" when seen in context.
            raw = TOKENIZER.decode(generated, skip_special_tokens=False)
            clean = _clean_output(raw)

            # Update the assistant message in history with current content
            # This ensures partial responses are saved if client disconnects
            assistant_msg["content"] = clean

            # The new visible text is whatever was added since last time
            delta = clean[prev_clean_len:]
            if delta:
                prev_clean_len = len(clean)
                yield _make_chunk(cid, delta)

    except Exception as e:
        # Send error as an SSE event so the client can display it
        err = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(err)}\n\n"
        # Update history with partial content even on error
        if generated:
            assistant_msg["content"] = _clean_output(TOKENIZER.decode(generated, skip_special_tokens=False))

    # Always send [DONE] to signal end of stream
    yield _make_done_chunk(cid)
