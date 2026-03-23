import importlib.util
from pathlib import Path
import sys
import time


def _load_scheduler_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "python" / "llaisys" / "scheduler.py"
    spec = importlib.util.spec_from_file_location("llaisys.scheduler", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load scheduler module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Svc:
    def __init__(self, name):
        self.name = name
        self.stop_calls = []

    def generate(self, payload):
        sid = str(payload.get("session_id") or "")
        return {
            "id": f"chatcmpl-{sid}",
            "object": "chat.completion",
            "model": "qwen2",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
            "session_id": sid,
            "worker": self.name,
        }

    def stream(self, payload):
        sid = str(payload.get("session_id") or "")
        yield {
            "id": f"chatcmpl-{sid}",
            "object": "chat.completion.chunk",
            "model": "qwen2",
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}],
            "session_id": sid,
        }
        yield {
            "id": f"chatcmpl-{sid}",
            "object": "chat.completion.chunk",
            "model": "qwen2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "session_id": sid,
        }

    def request_stop(self, session_id):
        self.stop_calls.append(session_id)
        return True

    def kv_debug_snapshot(self, session_id=None):
        return {"session_id": session_id, "has_native_context": False, "last_bind": {}, "kv_pool": {}}


class _SlowSvc(_Svc):
    def generate(self, payload):
        time.sleep(0.2)
        return super().generate(payload)


class _PackedSvc(_Svc):
    def __init__(self, name):
        super().__init__(name)
        self.packed_calls = 0

    def generate_packed_once(self, payloads):
        self.packed_calls += 1
        out = []
        for payload in payloads:
            sid = str(payload.get("session_id") or "")
            out.append({
                "id": f"chatcmpl-{sid}",
                "object": "chat.completion",
                "model": "qwen2",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "p"}, "finish_reason": "stop"}],
                "session_id": sid,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        return out

    def generate_packed_non_stream(self, payloads):
        return self.generate_packed_once(payloads)


def test_scheduler_non_stream_and_stream():
    mod = _load_scheduler_module()
    scheduler = mod.InferenceScheduler([_Svc("w0")], queue_size=4)
    scheduler.start()
    try:
        h1 = scheduler.submit({"session_id": "s1"}, stream=False)
        r1 = h1.get_result(timeout=2.0)
        assert r1["session_id"] == "s1"
        assert r1["worker"] == "w0"

        h2 = scheduler.submit({"session_id": "s1"}, stream=True)
        items = list(h2.iter_stream())
        assert items[-1]["choices"][0]["finish_reason"] is not None
        assert items[0]["choices"][0]["delta"]["content"] == "x"
    finally:
        scheduler.stop()


def test_scheduler_session_sticky_stop_route():
    mod = _load_scheduler_module()
    s0 = _Svc("w0")
    s1 = _Svc("w1")
    scheduler = mod.InferenceScheduler([s0, s1], queue_size=4)
    scheduler.start()
    try:
        # First bind session s-stick to a worker.
        h = scheduler.submit({"session_id": "s-stick"}, stream=False)
        _ = h.get_result(timeout=2.0)
        ok = scheduler.request_stop("s-stick")
        assert ok is True
        # Should only call one worker for mapped session.
        total = len(s0.stop_calls) + len(s1.stop_calls)
        assert total == 1
    finally:
        scheduler.stop()


def test_scheduler_queue_full_and_timeout():
    mod = _load_scheduler_module()
    scheduler = mod.InferenceScheduler([_SlowSvc("w0")], queue_size=1, request_timeout_ms=50)
    try:
        # Fill queue with first task.
        h1 = scheduler.submit({"session_id": "s-a"}, stream=False)
        # Second submit should fail due to queue full.
        try:
            scheduler.submit({"session_id": "s-b"}, stream=False)
            raise AssertionError("expected queue full")
        except mod.SchedulerQueueFullError:
            pass
        time.sleep(0.1)
        scheduler.start()
        # First task should timeout in worker before execution.
        r1 = h1.get_result(timeout=1.0)
        assert r1.get("code") == "timeout"
    finally:
        scheduler.stop()


def test_scheduler_continuous_batching_non_stream_path():
    mod = _load_scheduler_module()
    scheduler = mod.InferenceScheduler([_Svc("w0")], queue_size=4, request_timeout_ms=1000, continuous_batching=True)
    scheduler.start()
    try:
        h = scheduler.submit({"session_id": "s-cb"}, stream=False)
        r = h.get_result(timeout=2.0)
        assert r["session_id"] == "s-cb"
        assert "choices" in r
        assert r["choices"][0]["message"]["content"] is not None
        snap = scheduler.debug_snapshot()
        assert snap["continuous_batching"] is True
        assert snap["metrics"]["batch_rounds"] >= 1.0
        assert snap["metrics"]["prefill_rounds"] >= 1.0
        assert snap["metrics"]["decode_rounds"] >= 1.0
    finally:
        scheduler.stop()


def test_scheduler_continuous_batching_packed_prefill_path():
    mod = _load_scheduler_module()
    svc = _PackedSvc("w0")
    scheduler = mod.InferenceScheduler([svc], queue_size=8, request_timeout_ms=1000, continuous_batching=True)
    scheduler.start()
    try:
        h1 = scheduler.submit({"session_id": "a", "max_new_tokens": 1}, stream=False)
        h2 = scheduler.submit({"session_id": "b", "max_new_tokens": 1}, stream=False)
        r1 = h1.get_result(timeout=2.0)
        r2 = h2.get_result(timeout=2.0)
        assert r1["choices"][0]["message"]["content"] == "p"
        assert r2["choices"][0]["message"]["content"] == "p"
        snap = scheduler.debug_snapshot()
        assert snap["metrics"]["packed_prefill_batches"] >= 1.0
        assert snap["metrics"]["packed_prefill_tasks"] >= 2.0
        assert svc.packed_calls >= 1
    finally:
        scheduler.stop()


if __name__ == "__main__":
    test_scheduler_non_stream_and_stream()
    test_scheduler_session_sticky_stop_route()
    test_scheduler_queue_full_and_timeout()
    test_scheduler_continuous_batching_non_stream_path()
    test_scheduler_continuous_batching_packed_prefill_path()
    print("scheduler tests passed")
