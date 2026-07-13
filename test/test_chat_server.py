import json

from fastapi.testclient import TestClient

from llaisys.chat.server import ChatRuntime, create_app


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join([f"{m['role']}:{m['content']}\n" for m in messages]) + "assistant:"

    def encode(self, text):
        return [ord(c) % 256 for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(t) for t in tokens)


class _DummyModel:
    def generate(self, inputs, max_new_tokens, top_k, top_p, temperature):
        # Return prompt + "ok".
        return list(inputs) + [ord("o"), ord("k")]

    def generate_stream(self, inputs, max_new_tokens, top_k, top_p, temperature):
        yield ord("o")
        yield ord("k")


def test_chat_completion_response_shape():
    runtime = ChatRuntime(tokenizer=_DummyTokenizer(), model=_DummyModel())
    client = TestClient(create_app(runtime))

    payload = {
        "model": "qwen2",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "ok"
    assert body["usage"]["completion_tokens"] == 2


def test_chat_completion_stream_sse():
    runtime = ChatRuntime(tokenizer=_DummyTokenizer(), model=_DummyModel())
    client = TestClient(create_app(runtime))

    payload = {
        "model": "qwen2",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line]

    data_lines = [line for line in lines if line.startswith("data: ")]
    assert data_lines[-1] == "data: [DONE]"

    first_event = json.loads(data_lines[0][6:])
    assert first_event["choices"][0]["delta"]["role"] == "assistant"

    chunk_events = [json.loads(line[6:]) for line in data_lines[1:-1]]
    merged = "".join(e["choices"][0]["delta"].get("content", "") for e in chunk_events)
    assert merged == "ok"
