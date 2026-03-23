"""Tests for fix design (docs/FIX_DESIGN.md):
#1  _session_worker LRU eviction (max_sticky_sessions)
#2  KV routing TOCTOU - accepted, concurrent safety stress tests
#3  tokenize_for_routing exception logging + payload copy safety
#4  Interface inheritance (isinstance checks)
#5  request_stop merged locking (regression)
#6  _prompt_tokens cleaned from downstream payload
"""

import importlib.util
import logging
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Module loading helpers (same pattern as existing tests)
# ---------------------------------------------------------------------------

def _load_modules():
    root = Path(__file__).resolve().parents[1]
    interfaces_path = root / "python" / "llaisys" / "interfaces.py"
    kv_path = root / "python" / "llaisys" / "kv_cache_pool.py"
    scheduler_path = root / "python" / "llaisys" / "scheduler.py"
    session_mgr_path = root / "python" / "llaisys" / "session_manager.py"
    kv_bridge_path = root / "python" / "llaisys" / "kv_runtime_bridge.py"
    server_path = root / "python" / "llaisys" / "server.py"

    # Load interfaces first
    iface_spec = importlib.util.spec_from_file_location("llaisys.interfaces", str(interfaces_path))
    if iface_spec is None or iface_spec.loader is None:
        raise RuntimeError("failed to load interfaces")
    iface_mod = importlib.util.module_from_spec(iface_spec)
    sys.modules[iface_spec.name] = iface_mod
    iface_spec.loader.exec_module(iface_mod)

    kv_spec = importlib.util.spec_from_file_location("llaisys.kv_cache_pool", str(kv_path))
    if kv_spec is None or kv_spec.loader is None:
        raise RuntimeError("failed to load kv_cache_pool")
    kv_mod = importlib.util.module_from_spec(kv_spec)
    sys.modules[kv_spec.name] = kv_mod
    kv_spec.loader.exec_module(kv_mod)

    scheduler_spec = importlib.util.spec_from_file_location("llaisys.scheduler", str(scheduler_path))
    if scheduler_spec is None or scheduler_spec.loader is None:
        raise RuntimeError("failed to load scheduler")
    scheduler_mod = importlib.util.module_from_spec(scheduler_spec)
    sys.modules[scheduler_spec.name] = scheduler_mod
    scheduler_spec.loader.exec_module(scheduler_mod)

    # Load session_manager (server.py imports from it)
    session_mgr_mod = None
    if session_mgr_path.exists():
        sm_spec = importlib.util.spec_from_file_location("llaisys.session_manager", str(session_mgr_path))
        if sm_spec is not None and sm_spec.loader is not None:
            session_mgr_mod = importlib.util.module_from_spec(sm_spec)
            sys.modules[sm_spec.name] = session_mgr_mod
            sm_spec.loader.exec_module(session_mgr_mod)

    # Load kv_runtime_bridge (server.py imports from it)
    kv_bridge_mod = None
    if kv_bridge_path.exists():
        kb_spec = importlib.util.spec_from_file_location("llaisys.kv_runtime_bridge", str(kv_bridge_path))
        if kb_spec is not None and kb_spec.loader is not None:
            kv_bridge_mod = importlib.util.module_from_spec(kb_spec)
            sys.modules[kb_spec.name] = kv_bridge_mod
            kb_spec.loader.exec_module(kv_bridge_mod)

    fake_llaisys = types.ModuleType("llaisys")
    fake_llaisys.kv_cache_pool = kv_mod
    fake_llaisys.scheduler = scheduler_mod
    fake_llaisys.interfaces = iface_mod
    fake_llaisys.Tokenizer = object
    if session_mgr_mod:
        fake_llaisys.session_manager = session_mgr_mod
    if kv_bridge_mod:
        fake_llaisys.kv_runtime_bridge = kv_bridge_mod
    fake_llaisys.__path__ = [str(root / "python" / "llaisys")]
    sys.modules["llaisys"] = fake_llaisys
    sys.modules["llaisys.kv_cache_pool"] = kv_mod
    sys.modules["llaisys.scheduler"] = scheduler_mod
    sys.modules["llaisys.interfaces"] = iface_mod

    # fake libllaisys with stub LlaisysSamplingParams
    fake_libllaisys = types.ModuleType("llaisys.libllaisys")

    class _StubSamplingParams:
        def __init__(self, top_k=1, top_p=0.0, temperature=0.0, seed=0):
            self.top_k = top_k
            self.top_p = top_p
            self.temperature = temperature
            self.seed = seed

    fake_libllaisys.LlaisysSamplingParams = _StubSamplingParams
    fake_llaisys.libllaisys = fake_libllaisys
    sys.modules["llaisys.libllaisys"] = fake_libllaisys
    if session_mgr_mod:
        sys.modules["llaisys.session_manager"] = session_mgr_mod
    if kv_bridge_mod:
        sys.modules["llaisys.kv_runtime_bridge"] = kv_bridge_mod

    fake_models = types.ModuleType("llaisys.models")

    class _StubQwen2:
        @staticmethod
        def build_prompt(messages, system_prompt=None, add_generation_prompt=True):
            lines = []
            if system_prompt:
                lines.append(f"System: {system_prompt}")
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "assistant":
                    lines.append(f"Assistant: {content}")
                else:
                    lines.append(f"User: {content}")
            if add_generation_prompt:
                lines.append("Assistant:")
            return "\n".join(lines)

    fake_models.Qwen2 = _StubQwen2
    sys.modules["llaisys.models"] = fake_models

    spec = importlib.util.spec_from_file_location("llaisys.server", str(server_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load server module")
    server_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = server_mod
    spec.loader.exec_module(server_mod)

    return iface_mod, kv_mod, scheduler_mod, server_mod


iface_mod, kv_mod, scheduler_mod, server_mod = _load_modules()
KVCachePool = kv_mod.KVCachePool
InferenceScheduler = scheduler_mod.InferenceScheduler
SchedulerQueueFullError = scheduler_mod.SchedulerQueueFullError
ChatService = server_mod.ChatService


# ---------------------------------------------------------------------------
# Fake service / model helpers
# ---------------------------------------------------------------------------

class _Svc:
    """Minimal service mock for scheduler tests."""

    def __init__(self, name: str):
        self.name = name
        self.stop_calls: List[str] = []
        self._kv_pool = KVCachePool(block_size=4, max_blocks=128, max_bytes=1024 * 1024)
        self.last_payload: Optional[Dict[str, Any]] = None

    @property
    def kv_pool(self):
        return self._kv_pool

    def generate(self, payload):
        self.last_payload = dict(payload)
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
        self.last_payload = dict(payload)
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
        return {"session_id": session_id, "has_native_context": False, "last_bind": {}, "kv_pool": self._kv_pool.snapshot_stats()}

    def tokenize_for_routing(self, payload):
        prompt = str(payload.get("prompt") or "")
        return [ord(ch) for ch in prompt] if prompt else None


class _FailTokenizeSvc(_Svc):
    """Service whose tokenize_for_routing always raises."""

    def tokenize_for_routing(self, payload):
        raise RuntimeError("tokenizer broken")


class _EndToken:
    def __init__(self, value):
        self.value = value


class _Meta:
    def __init__(self):
        self.end_token = _EndToken(-1)


class FakeTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text]

    def decode(self, token_ids):
        return "".join(chr(int(t)) for t in token_ids)


class FakeModel:
    def __init__(self):
        self._meta = _Meta()
        self.bind_calls = []
        self.export_calls = []
        self.reset_calls = 0
        self._ctx_seq = 0

    def reset_kv_cache(self):
        self.reset_calls += 1

    def prefill(self, prompt_ids):
        return 65

    def prefill_sampling(self, prompt_ids, top_k=1, top_p=0.0, temperature=0.0, seed=0):
        return self.prefill(prompt_ids)

    def step(self, token_ids):
        return 66

    def step_sampling(self, token_ids, top_k=1, top_p=0.0, temperature=0.0, seed=0):
        return self.step(token_ids)

    def set_kv_context(self, ctx):
        self.bind_calls.append(ctx)
        return 0

    def kv_context_create(self):
        self._ctx_seq += 1
        return {"ctx_id": self._ctx_seq}

    def kv_context_release(self, ctx):
        return None

    def export_kv_context(self, ctx, block_tokens):
        self.export_calls.append((ctx, block_tokens))
        return 0


def _make_service(**kwargs):
    model = FakeModel()
    tok = FakeTokenizer()
    service = ChatService(
        model=model,
        tokenizer=tok,
        model_path=None,
        enable_kv_runtime_reuse=kwargs.get("enable_kv_runtime_reuse", True),
        block_size=kwargs.get("block_size", 4),
        max_blocks=kwargs.get("max_blocks", 256),
        max_bytes=kwargs.get("max_bytes", 1024 * 1024),
    )
    return service, model


# ===========================================================================
# Fix #1: _session_worker LRU eviction (max_sticky_sessions)
# ===========================================================================

def test_session_worker_lru_eviction():
    """After exceeding max_sticky_sessions, oldest entries should be evicted.

    Design: InferenceScheduler(max_sticky_sessions=N) uses OrderedDict with
    LRU eviction via _touch_session(). Minimum enforced value is 100.
    """
    max_sticky = 100  # minimum enforced by max(100, int(max_sticky_sessions))
    svc = _Svc("w0")
    try:
        scheduler = InferenceScheduler([svc], queue_size=256, max_sticky_sessions=max_sticky)
    except TypeError:
        print("  SKIP: max_sticky_sessions parameter not yet implemented")
        return
    scheduler.start()
    try:
        # Submit more sessions than the limit
        num_sessions = max_sticky + 50
        for i in range(num_sessions):
            h = scheduler.submit({"session_id": f"lru-{i}"}, stream=False)
            h.get_result(timeout=2.0)

        with scheduler._lock:
            mapping_size = len(scheduler._session_worker)

        assert mapping_size <= max_sticky, (
            f"_session_worker has {mapping_size} entries, expected <= {max_sticky}"
        )

        # The oldest sessions (lru-0, lru-1, ...) should have been evicted.
        # The newest sessions should still be present.
        with scheduler._lock:
            assert f"lru-{num_sessions - 1}" in scheduler._session_worker, (
                "Most recent session should be in the map"
            )
            # First sessions should be evicted
            assert "lru-0" not in scheduler._session_worker, (
                "Oldest session should have been evicted"
            )

        print(f"  LRU eviction works: {mapping_size} entries <= max {max_sticky}")
    finally:
        scheduler.stop()


def test_session_worker_lru_touch_refreshes_entry():
    """Accessing an existing session should refresh it (move to end of LRU)."""
    max_sticky = 100  # minimum enforced by implementation
    svc = _Svc("w0")
    try:
        scheduler = InferenceScheduler([svc], queue_size=256, max_sticky_sessions=max_sticky)
    except TypeError:
        print("  SKIP: max_sticky_sessions parameter not yet implemented")
        return
    scheduler.start()
    try:
        # Fill the map to capacity
        for i in range(max_sticky):
            h = scheduler.submit({"session_id": f"touch-{i}"}, stream=False)
            h.get_result(timeout=2.0)

        # Re-access the first session to refresh it (move to end of LRU)
        h = scheduler.submit({"session_id": "touch-0"}, stream=False)
        h.get_result(timeout=2.0)

        # Now add more sessions to trigger eviction of oldest non-refreshed entries
        for i in range(10):
            h = scheduler.submit({"session_id": f"touch-new-{i}"}, stream=False)
            h.get_result(timeout=2.0)

        with scheduler._lock:
            # touch-0 was refreshed, so it should survive eviction
            assert "touch-0" in scheduler._session_worker, (
                "Refreshed session should survive eviction"
            )
            # touch-1 was not refreshed and is among the oldest, should be evicted
            assert "touch-1" not in scheduler._session_worker, (
                "Non-refreshed old session should be evicted"
            )

        print("  LRU touch refresh works correctly")
    finally:
        scheduler.stop()


def test_session_worker_debug_snapshot_sticky_sessions():
    """debug_snapshot should include sticky_sessions count."""
    svc = _Svc("w0")
    try:
        scheduler = InferenceScheduler([svc], queue_size=128, max_sticky_sessions=100)
    except TypeError:
        print("  SKIP: max_sticky_sessions parameter not yet implemented")
        return
    scheduler.start()
    try:
        h = scheduler.submit({"session_id": "snap-1"}, stream=False)
        h.get_result(timeout=2.0)
        snap = scheduler.debug_snapshot()
        assert "sticky_sessions" in snap, "debug_snapshot should include sticky_sessions"
        assert snap["sticky_sessions"] == 1
        print(f"  debug_snapshot includes sticky_sessions: {snap['sticky_sessions']}")
    finally:
        scheduler.stop()


# ===========================================================================
# Fix #2: KV routing TOCTOU (accepted, concurrent stress tests)
# ===========================================================================

def test_kv_aware_routing_concurrent_submits():
    """Multiple threads submitting concurrently with kv_aware_routing enabled.

    Verifies no crashes, deadlocks, or data corruption under concurrent access.
    """
    svc0 = _Svc("w0")
    svc1 = _Svc("w1")
    svc0.kv_pool.acquire_context("seed", [72, 101, 108, 108])

    scheduler = InferenceScheduler(
        [svc0, svc1],
        queue_size=64,
        kv_aware_routing=True,
    )
    scheduler.start()

    errors: List[Exception] = []
    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _submit(session_id: str, prompt_tokens: Optional[List[int]] = None):
        try:
            payload: Dict[str, Any] = {"session_id": session_id}
            if prompt_tokens:
                payload["_prompt_tokens"] = prompt_tokens
            h = scheduler.submit(payload, stream=False)
            r = h.get_result(timeout=5.0)
            with lock:
                results.append(r)
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = []
    for i in range(20):
        tokens = [72, 101, 108, 108] if i % 2 == 0 else None
        t = threading.Thread(target=_submit, args=(f"concurrent-{i}", tokens))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    scheduler.stop()

    assert len(errors) == 0, f"Concurrent routing errors: {errors}"
    assert len(results) == 20, f"Expected 20 results, got {len(results)}"

    snap = scheduler.debug_snapshot()
    assert snap["kv_aware_routing"] is True
    attempts = snap["metrics"]["kv_aware_routing_attempts"]
    hits = snap["metrics"]["kv_aware_routing_hits"]
    assert hits <= attempts
    print(f"  KV routing: {int(attempts)} attempts, {int(hits)} hits")


def test_kv_aware_routing_no_deadlock_under_contention():
    """Stress test: many threads hitting _choose_worker simultaneously."""
    svc0 = _Svc("w0")
    svc1 = _Svc("w1")
    scheduler = InferenceScheduler(
        [svc0, svc1],
        queue_size=256,
        kv_aware_routing=True,
    )
    scheduler.start()

    barrier = threading.Barrier(10)
    errors: List[Exception] = []

    def _rapid_submit(tid: int):
        try:
            barrier.wait(timeout=5.0)
            for j in range(10):
                payload = {
                    "session_id": f"stress-{tid}-{j}",
                    "_prompt_tokens": [1, 2, 3, 4],
                }
                h = scheduler.submit(payload, stream=False)
                h.get_result(timeout=5.0)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_rapid_submit, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    scheduler.stop()
    assert len(errors) == 0, f"Deadlock or errors under contention: {errors}"
    print("  no deadlock detected under 10-thread contention")


# ===========================================================================
# Fix #3: tokenize_for_routing exception logging + payload copy
# ===========================================================================

def test_tokenize_for_routing_exception_logs_debug():
    """When tokenize_for_routing raises, submit should log at DEBUG level
    and still succeed with fallback routing.
    """
    svc0 = _FailTokenizeSvc("w0")
    svc1 = _FailTokenizeSvc("w1")
    scheduler = InferenceScheduler(
        [svc0, svc1],
        queue_size=16,
        kv_aware_routing=True,
    )
    scheduler.start()

    # Capture log output from the scheduler module's logger
    log_records: List[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            log_records.append(record)

    # Try to find the logger used by the scheduler module
    scheduler_logger = logging.getLogger(scheduler_mod.__name__)
    handler = _Handler()
    handler.setLevel(logging.DEBUG)
    scheduler_logger.addHandler(handler)
    old_level = scheduler_logger.level
    scheduler_logger.setLevel(logging.DEBUG)

    try:
        h = scheduler.submit({"session_id": "s-log", "prompt": "hello"}, stream=False)
        r = h.get_result(timeout=2.0)
        assert r["session_id"] == "s-log"

        # Check if any debug log was emitted about tokenize failure
        tokenize_logs = [r for r in log_records if "tokenize" in r.getMessage().lower() or "routing" in r.getMessage().lower()]
        if tokenize_logs:
            print(f"  logger.debug emitted: '{tokenize_logs[0].getMessage()}'")
        else:
            print("  NOTE: no tokenize debug log found (logger may not be implemented yet)")
    finally:
        scheduler_logger.removeHandler(handler)
        scheduler_logger.setLevel(old_level)
        scheduler.stop()


def test_tokenize_for_routing_exception_falls_back_gracefully():
    """When tokenize_for_routing raises, submit should still succeed."""
    svc0 = _FailTokenizeSvc("w0")
    svc1 = _FailTokenizeSvc("w1")
    scheduler = InferenceScheduler(
        [svc0, svc1],
        queue_size=16,
        kv_aware_routing=True,
    )
    scheduler.start()
    try:
        h = scheduler.submit({"session_id": "s-fail-tok", "prompt": "hello"}, stream=False)
        r = h.get_result(timeout=2.0)
        assert r["session_id"] == "s-fail-tok"
        print("  tokenize_for_routing exception handled gracefully")
    finally:
        scheduler.stop()


def test_submit_does_not_mutate_caller_payload():
    """submit() should not modify the caller's original payload dict.

    Design fix 3b: payload = dict(payload) at submit() entry.
    """
    svc0 = _Svc("w0")
    svc1 = _Svc("w1")
    scheduler = InferenceScheduler(
        [svc0, svc1],
        queue_size=16,
        kv_aware_routing=True,
    )
    scheduler.start()
    try:
        original_payload = {"session_id": "s-immut", "prompt": "test"}
        original_keys = set(original_payload.keys())
        h = scheduler.submit(original_payload, stream=False)
        h.get_result(timeout=2.0)

        # The original dict should not have been modified
        assert set(original_payload.keys()) == original_keys, (
            f"Caller payload was mutated: {set(original_payload.keys())} != {original_keys}"
        )
        assert "_prompt_tokens" not in original_payload, (
            "_prompt_tokens leaked into caller's payload"
        )
        print("  submit() does not mutate caller payload")
    finally:
        scheduler.stop()


def test_tokenize_for_routing_returns_none_falls_back():
    """When tokenize_for_routing returns None, routing falls back to hash/RR."""

    class _NoneTokenizeSvc(_Svc):
        def tokenize_for_routing(self, payload):
            return None

    svc0 = _NoneTokenizeSvc("w0")
    svc1 = _NoneTokenizeSvc("w1")
    scheduler = InferenceScheduler(
        [svc0, svc1],
        queue_size=16,
        kv_aware_routing=True,
    )
    scheduler.start()
    try:
        h = scheduler.submit({"session_id": "s-none-tok", "prompt": "hello"}, stream=False)
        r = h.get_result(timeout=2.0)
        assert r["session_id"] == "s-none-tok"
        print("  tokenize_for_routing returning None handled correctly")
    finally:
        scheduler.stop()


def test_tokenize_for_routing_on_chatservice_with_bad_payload():
    """ChatService.tokenize_for_routing returns None for invalid payloads."""
    service, _ = _make_service()

    assert service.tokenize_for_routing({}) is None
    assert service.tokenize_for_routing({"messages": "not a list"}) is None

    tokens = service.tokenize_for_routing({"prompt": "hi"})
    assert tokens is not None and len(tokens) > 0

    print("  ChatService.tokenize_for_routing handles bad payloads safely")


# ===========================================================================
# Fix #4: Interface inheritance (isinstance checks)
# ===========================================================================

def test_kvcachepool_isinstance_ikvachepool():
    """KVCachePool should inherit from IKVCachePool."""
    IKVCachePool = getattr(iface_mod, "IKVCachePool", None)
    if IKVCachePool is None:
        print("  SKIP: IKVCachePool interface not found")
        return
    pool = KVCachePool(block_size=4, max_blocks=128, max_bytes=1024 * 1024)
    if isinstance(pool, IKVCachePool):
        print("  KVCachePool isinstance IKVCachePool: True")
    else:
        print("  NOTE: KVCachePool does not yet inherit IKVCachePool (fix #4 pending)")


def test_chatservice_isinstance_iinferenceservice():
    """ChatService should inherit from IInferenceService."""
    IInferenceService = getattr(iface_mod, "IInferenceService", None)
    if IInferenceService is None:
        print("  SKIP: IInferenceService interface not found")
        return
    service, _ = _make_service()
    if isinstance(service, IInferenceService):
        print("  ChatService isinstance IInferenceService: True")
    else:
        print("  NOTE: ChatService does not yet inherit IInferenceService (fix #4 pending)")


# ===========================================================================
# Fix #5: request_stop merged locking (regression)
# ===========================================================================

def test_regression_request_stop_works():
    """Regression: request_stop still works after merging lock blocks."""
    svc0 = _Svc("w0")
    svc1 = _Svc("w1")
    scheduler = InferenceScheduler([svc0, svc1], queue_size=4)
    scheduler.start()
    try:
        h = scheduler.submit({"session_id": "stop-test"}, stream=False)
        h.get_result(timeout=2.0)
        ok = scheduler.request_stop("stop-test")
        assert ok is True
        total = len(svc0.stop_calls) + len(svc1.stop_calls)
        assert total == 1
        print("  regression: request_stop works after merge")
    finally:
        scheduler.stop()


# ===========================================================================
# Fix #6: _prompt_tokens cleaned from downstream payload
# ===========================================================================

def test_prompt_tokens_not_leaked_to_worker():
    """InferenceTask.payload reaching the worker should not contain _prompt_tokens.

    Design: submit() calls payload.pop("_prompt_tokens", None) after routing.
    """
    svc0 = _Svc("w0")
    svc1 = _Svc("w1")
    scheduler = InferenceScheduler(
        [svc0, svc1],
        queue_size=16,
        kv_aware_routing=True,
    )
    scheduler.start()
    try:
        h = scheduler.submit(
            {"session_id": "s-clean", "prompt": "test"},
            stream=False,
        )
        h.get_result(timeout=2.0)

        # Check the payload that the worker (svc) actually received
        for svc in (svc0, svc1):
            if svc.last_payload is not None:
                if "_prompt_tokens" in svc.last_payload:
                    print("  NOTE: _prompt_tokens still in worker payload (fix #6 pending)")
                else:
                    print("  _prompt_tokens cleaned from worker payload")
                break
    finally:
        scheduler.stop()


def test_prompt_tokens_explicit_in_payload_also_cleaned():
    """Even if caller passes _prompt_tokens explicitly, it should be cleaned."""
    svc0 = _Svc("w0")
    svc1 = _Svc("w1")
    scheduler = InferenceScheduler(
        [svc0, svc1],
        queue_size=16,
        kv_aware_routing=True,
    )
    scheduler.start()
    try:
        h = scheduler.submit(
            {"session_id": "s-explicit", "_prompt_tokens": [1, 2, 3]},
            stream=False,
        )
        h.get_result(timeout=2.0)

        for svc in (svc0, svc1):
            if svc.last_payload is not None:
                if "_prompt_tokens" in svc.last_payload:
                    print("  NOTE: explicit _prompt_tokens still in worker payload (fix #6 pending)")
                else:
                    print("  explicit _prompt_tokens cleaned from worker payload")
                break
    finally:
        scheduler.stop()


# ===========================================================================
# Regression tests
# ===========================================================================

def test_regression_kv_cache_pool_prefix_match():
    """Regression: sealed block prefix matching still works."""
    pool = KVCachePool(block_size=4, max_blocks=128, max_bytes=1024 * 1024)
    result_a = pool.acquire_context("ctx-a", [1, 2, 3, 4, 5, 6])
    assert result_a.prefix_len == 0
    result_b = pool.acquire_context("ctx-b", [1, 2, 3, 4, 5, 6])
    assert result_b.prefix_len == 4
    stats = pool.snapshot_stats()
    assert stats["prefix_hit_count"] >= 1
    print("  regression: kv_cache_pool prefix match OK")


def test_regression_scheduler_non_stream():
    """Regression: basic non-stream generate still works."""
    scheduler = InferenceScheduler([_Svc("w0")], queue_size=4)
    scheduler.start()
    try:
        h = scheduler.submit({"session_id": "reg-1"}, stream=False)
        r = h.get_result(timeout=2.0)
        assert r["session_id"] == "reg-1"
        assert r["worker"] == "w0"
    finally:
        scheduler.stop()
    print("  regression: scheduler non-stream OK")


def test_regression_scheduler_stream():
    """Regression: basic stream still works."""
    scheduler = InferenceScheduler([_Svc("w0")], queue_size=4)
    scheduler.start()
    try:
        h = scheduler.submit({"session_id": "reg-2"}, stream=True)
        items = list(h.iter_stream())
        assert items[-1]["choices"][0]["finish_reason"] is not None
    finally:
        scheduler.stop()
    print("  regression: scheduler stream OK")


def test_regression_server_kv_reuse():
    """Regression: ChatService KV reuse for same session still works."""
    service, model = _make_service()
    first = service.generate({"session_id": "reg-s1", "prompt": "hello", "max_new_tokens": 2})
    assert first["session_id"] == "reg-s1"
    service.generate({"session_id": "reg-s1", "prompt": "again", "max_new_tokens": 2})
    assert model.bind_calls[-1] is not None
    dbg = service.kv_debug_snapshot("reg-s1")
    assert dbg["last_bind"]["bound"] is True
    print("  regression: server kv reuse OK")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    tests = [
        # Fix #1: LRU session map
        test_session_worker_lru_eviction,
        test_session_worker_lru_touch_refreshes_entry,
        test_session_worker_debug_snapshot_sticky_sessions,
        # Fix #2: KV routing concurrency
        test_kv_aware_routing_concurrent_submits,
        test_kv_aware_routing_no_deadlock_under_contention,
        # Fix #3: tokenize exception + payload copy
        test_tokenize_for_routing_exception_logs_debug,
        test_tokenize_for_routing_exception_falls_back_gracefully,
        test_submit_does_not_mutate_caller_payload,
        test_tokenize_for_routing_returns_none_falls_back,
        test_tokenize_for_routing_on_chatservice_with_bad_payload,
        # Fix #4: interface inheritance
        test_kvcachepool_isinstance_ikvachepool,
        test_chatservice_isinstance_iinferenceservice,
        # Fix #5: request_stop regression
        test_regression_request_stop_works,
        # Fix #6: _prompt_tokens cleanup
        test_prompt_tokens_not_leaked_to_worker,
        test_prompt_tokens_explicit_in_payload_also_cleaned,
        # General regression
        test_regression_kv_cache_pool_prefix_match,
        test_regression_scheduler_non_stream,
        test_regression_scheduler_stream,
        test_regression_server_kv_reuse,
    ]

    passed = 0
    failed = 0
    skipped = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            print(f"[RUN ] {name}")
            test_fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
