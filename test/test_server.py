import importlib
import json
import os
import sys
import uuid

from fastapi.testclient import TestClient


def _load_app_with_real_engine():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_root = os.path.join(repo_root, "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)

    if "server.app" in sys.modules:
        del sys.modules["server.app"]
    if "server.engine" in sys.modules:
        del sys.modules["server.engine"]

    return importlib.import_module("server.app")


def _collect_stream_reply(chunks):
    parts = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            try:
                item = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = item.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                parts.append(piece)
    return "".join(parts)


def _print_prompt_debug(app_module, messages, label):
    backend = app_module.ENGINE._backend
    prompt = backend._build_prompt(messages)
    print(f"[{label} messages repr] {ascii(messages)}")
    print(f"[{label} prompt repr] {ascii(prompt)}")
    print(f"[{label} messages has <think>] {any('<think>' in (m.get('content') or '') or '</think>' in (m.get('content') or '') for m in messages)}")
    print(f"[{label} prompt has <think>] {'<think>' in prompt}")
    print(f"[{label} prompt has </think>] {'</think>' in prompt}")


def test_healthz(app_module=None):
    
    client = TestClient(app_module.app)

    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert "model" in body


def test_ui_index(app_module=None):
    client = TestClient(app_module.app)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "LLAISYS Chat UI" in resp.text


def test_chat_completion_non_stream(app_module=None):
    
    client = TestClient(app_module.app)

    payload = {
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False,
        "max_tokens": 64,
        "top_k": 10,
        "top_p": 0.9,
        "temperature": 0.7,
    }

    _print_prompt_debug(app_module, payload["messages"], "non-stream")

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    print(f"[non-stream reply] {ascii(body['choices'][0]['message']['content'])}")
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] >= 0
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]


def test_chat_completion_stream(app_module=None):
    client = TestClient(app_module.app)

    payload = {
        "messages": [{"role": "user", "content": "你好"}],
        "stream": True,
        "max_tokens": 64,
        "top_k": 10,
        "top_p": 0.9,
        "temperature": 0.7,
    }

    _print_prompt_debug(app_module, payload["messages"], "stream")

    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        chunks = list(resp.iter_text())

    merged = "".join(chunks)
    stream_reply = _collect_stream_reply(chunks)
    print(f"[stream reply] {ascii(stream_reply)}")
    assert "chat.completion.chunk" in merged
    assert '"finish_reason": "stop"' in merged
    assert "data: [DONE]" in merged
    assert isinstance(stream_reply, str)


def test_chat_completion_empty_messages(app_module=None):
    client = TestClient(app_module.app)

    payload = {
        "messages": [],
        "stream": False,
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400
    assert "messages cannot be empty" in resp.text


def test_session_apis_and_session_chat(app_module=None):
    client = TestClient(app_module.app)

    session_id = f"test-sess-{uuid.uuid4().hex[:8]}"

    create_resp = client.post("/v1/sessions", json={"session_id": session_id})
    assert create_resp.status_code == 200
    assert create_resp.json()["session_id"] == session_id

    list_resp = client.get("/v1/sessions")
    assert list_resp.status_code == 200
    assert session_id in list_resp.json().get("sessions", [])

    payload = {
        "session_id": session_id,
        "messages": [{"role": "user", "content": "who are you?"}],
        "stream": False,
        "max_tokens": 64,
    }
    chat_resp = client.post("/v1/chat/completions", json=payload)
    assert chat_resp.status_code == 200
    assert chat_resp.json()["object"] == "chat.completion"

    session_resp = client.get(f"/v1/sessions/{session_id}")
    assert session_resp.status_code == 200
    session_messages = session_resp.json().get("messages", [])
    assert any(item.get("role") == "assistant" for item in session_messages)

    delete_resp = client.delete(f"/v1/sessions/{session_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] == session_id


if __name__ == "__main__":
    app_module = _load_app_with_real_engine()
    test_healthz(app_module)
    test_ui_index(app_module)
    test_chat_completion_non_stream(app_module)
    test_chat_completion_stream(app_module)
    test_chat_completion_empty_messages(app_module)
    test_session_apis_and_session_chat(app_module)
    print("\033[92mTest passed!\033[0m\n")
