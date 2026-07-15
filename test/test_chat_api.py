import sys
import threading
import time


def _has_optional_deps() -> bool:
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


if not _has_optional_deps():
    print("Skipped: optional server dependencies are not installed.")
    sys.exit(0)

from fastapi.testclient import TestClient

from chat_test_utils import FakeModel, FakeTokenizer
from llaisys.chat_server import create_app


def test_non_stream_response() -> None:
    app = create_app(tokenizer=FakeTokenizer(), model=FakeModel(), model_name="fake-qwen")
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "top_k": 10,
            "top_p": 0.9,
            "temperature": 0.8,
        },
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "fake-qwen"
    assert payload["session_id"].startswith("session-")
    assert payload["cache_reused_tokens"] == 0
    assert payload["worker_id"] == 0
    assert payload["batch_id"] == 1
    assert payload["dispatch_count"] == 6
    assert payload["last_batch_id"] >= payload["batch_id"]
    assert payload["choices"][0]["message"]["content"] == "Hello!"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["completion_tokens"] == 6


def test_stream_response() -> None:
    app = create_app(tokenizer=FakeTokenizer(), model=FakeModel(), model_name="fake-qwen")
    client = TestClient(app)
    chunks = []
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "top_k": 10,
            "top_p": 0.9,
            "temperature": 0.8,
            "stream": True,
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                chunks.append(line)

    assert chunks[-1] == "data: [DONE]"
    content_parts = []
    for line in chunks[:-1]:
        if not line.startswith("data: "):
            continue
        event = line[6:]
        if event == "[DONE]":
            continue
        payload = __import__("json").loads(event)
        delta = payload["choices"][0]["delta"]
        if "content" in delta:
            content_parts.append(delta["content"])
    assert "".join(content_parts) == "Hello!"


def test_service_stats_and_multi_user_batch_dispatch() -> None:
    app = create_app(
        tokenizer=FakeTokenizer(),
        model_factory=lambda: FakeModel(delay_s=0.05),
        model_name="fake-qwen",
        num_workers=2,
        max_batch_size=2,
        batch_wait_ms=20,
    )
    client = TestClient(app)
    results = [None, None]

    def send(index: int, prompt: str) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-qwen",
                "session_id": f"session-batch-{index}",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8,
            },
        )
        response.raise_for_status()
        results[index] = response.json()

    t0 = threading.Thread(target=send, args=(0, "alpha"))
    t1 = threading.Thread(target=send, args=(1, "beta"))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    payload0 = results[0]
    payload1 = results[1]
    assert payload0 is not None and payload1 is not None
    assert payload0["choices"][0]["message"]["content"] == "Hello!"
    assert payload1["choices"][0]["message"]["content"] == "Hello!"
    assert payload0["batch_id"] == payload1["batch_id"]
    assert {payload0["worker_id"], payload1["worker_id"]} == {0, 1}
    assert payload0["dispatch_count"] == 6
    assert payload1["dispatch_count"] == 6

    stats = client.get("/v1/service/stats")
    stats.raise_for_status()
    body = stats.json()
    assert body["worker_count"] == 2
    assert body["scheduled_batches"] >= 6
    assert body["max_observed_batch_size"] >= 2
    assert body["completed_requests"] == 2
    assert body["queue_depth"] == 0
    assert body["requeued_requests"] >= 10
    assert body["total_generated_tokens"] == 12
    assert body["cached_session_count"] == 2


def test_same_session_concurrent_request_is_rejected() -> None:
    app = create_app(
        tokenizer=FakeTokenizer(),
        model_factory=lambda: FakeModel(delay_s=0.1),
        model_name="fake-qwen",
        num_workers=2,
        max_batch_size=2,
        batch_wait_ms=10,
    )
    client = TestClient(app)
    holder = {}

    def first_request() -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-qwen",
                "session_id": "session-busy",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
            },
        )
        holder["first"] = response

    thread = threading.Thread(target=first_request)
    thread.start()

    deadline = time.time() + 3.0
    while time.time() < deadline:
        stats = client.get("/v1/service/stats")
        stats.raise_for_status()
        if stats.json()["active_requests"] > 0:
            break
        time.sleep(0.01)

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": "session-busy",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
    )
    assert second.status_code == 409
    assert "in-flight request" in second.json()["detail"]

    thread.join()
    holder["first"].raise_for_status()


def test_session_reuse_response() -> None:
    model = FakeModel()
    app = create_app(tokenizer=FakeTokenizer(), model=model, model_name="fake-qwen")
    client = TestClient(app)
    session_id = "session-reuse"

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": session_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
    )
    first.raise_for_status()
    assistant_content = first.json()["choices"][0]["message"]["content"]

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": "again"},
            ],
            "max_tokens": 8,
        },
    )
    second.raise_for_status()
    payload = second.json()
    assert payload["cache_reused_tokens"] > 0
    generate_next_calls = [call for call in model.calls if call["mode"] == "generate_next"]
    assert generate_next_calls[0]["reset_state"] is True
    resumed_call = next(call for call in generate_next_calls[1:] if call["reset_state"] is False and len(call["inputs"]) > 1)
    assert resumed_call["reset_state"] is False
    assert len(resumed_call["inputs"]) < len(generate_next_calls[0]["inputs"]) + len(assistant_content)


def test_session_affinity_preserves_cache_slot_across_workers() -> None:
    app = create_app(
        tokenizer=FakeTokenizer(),
        model_factory=FakeModel,
        model_name="fake-qwen",
        num_workers=2,
        max_batch_size=2,
        batch_wait_ms=10,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": "session-affinity-a",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
    )
    first.raise_for_status()
    first_payload = first.json()

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": "session-affinity-b",
            "messages": [{"role": "user", "content": "beta"}],
            "max_tokens": 8,
        },
    )
    second.raise_for_status()
    second_payload = second.json()
    assert first_payload["worker_id"] != second_payload["worker_id"]

    assistant_content = first_payload["choices"][0]["message"]["content"]
    third = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": "session-affinity-a",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": "again"},
            ],
            "max_tokens": 8,
        },
    )
    third.raise_for_status()
    third_payload = third.json()
    assert third_payload["worker_id"] == first_payload["worker_id"]
    assert third_payload["cache_reused_tokens"] > 0


def test_session_management_crud() -> None:
    app = create_app(tokenizer=FakeTokenizer(), model=FakeModel(), model_name="fake-qwen")
    client = TestClient(app)

    created = client.post(
        "/v1/sessions",
        json={
            "session_id": "session-crud",
            "messages": [{"role": "system", "content": "You are terse."}],
        },
    )
    created.raise_for_status()
    assert created.json()["id"] == "session-crud"

    listed = client.get("/v1/sessions")
    listed.raise_for_status()
    session_ids = [item["id"] for item in listed.json()["data"]]
    assert "session-crud" in session_ids

    fetched = client.get("/v1/sessions/session-crud")
    fetched.raise_for_status()
    assert fetched.json()["messages"][0]["content"] == "You are terse."

    updated = client.put(
        "/v1/sessions/session-crud",
        json={"messages": [{"role": "user", "content": "edited"}]},
    )
    updated.raise_for_status()
    assert updated.json()["messages"][0]["content"] == "edited"

    deleted = client.delete("/v1/sessions/session-crud")
    deleted.raise_for_status()
    assert deleted.json()["deleted"] is True


def test_regenerate_uses_truncate_for_active_session() -> None:
    model = FakeModel()
    app = create_app(tokenizer=FakeTokenizer(), model=model, model_name="fake-qwen")
    client = TestClient(app)
    session_id = "session-regen"

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": session_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
    )
    first.raise_for_status()

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": session_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
    )
    second.raise_for_status()

    truncate_calls = [call for call in model.calls if call["mode"] == "truncate"]
    assert truncate_calls, model.calls
    assert truncate_calls[-1]["position"] > 0
    generate_next_calls = [call for call in model.calls if call["mode"] == "generate_next"]
    assert generate_next_calls[0]["reset_state"] is True
    truncate_index = max(idx for idx, call in enumerate(model.calls) if call["mode"] == "truncate")
    resumed_call = next(call for call in model.calls[truncate_index + 1:] if call["mode"] == "generate_next")
    assert resumed_call["reset_state"] is False


if __name__ == "__main__":
    test_non_stream_response()
    test_stream_response()
    test_service_stats_and_multi_user_batch_dispatch()
    test_same_session_concurrent_request_is_rejected()
    test_session_reuse_response()
    test_session_affinity_preserves_cache_slot_across_workers()
    test_session_management_crud()
    test_regenerate_uses_truncate_for_active_session()
    print("\033[92mTest passed!\033[0m\n")
