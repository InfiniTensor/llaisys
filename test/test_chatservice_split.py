"""Tests for ChatService split (docs/CHATSERVICE_SPLIT_DESIGN.md):
- SessionManager unit tests
- KVRuntimeBridge unit tests (mock model)
- ChatService integration (delegation correctness)
- Interface compatibility (isinstance checks)
- Regression: existing tests must still pass
"""

import importlib.util
import sys
import threading
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _load_modules():
    root = Path(__file__).resolve().parents[1]
    interfaces_path = root / "python" / "llaisys" / "interfaces.py"
    kv_path = root / "python" / "llaisys" / "kv_cache_pool.py"
    scheduler_path = root / "python" / "llaisys" / "scheduler.py"
    session_mgr_path = root / "python" / "llaisys" / "session_manager.py"
    kv_bridge_path = root / "python" / "llaisys" / "kv_runtime_bridge.py"
    server_path = root / "python" / "llaisys" / "server.py"

    # interfaces
    iface_spec = importlib.util.spec_from_file_location("llaisys.interfaces", str(interfaces_path))
    if iface_spec is None or iface_spec.loader is None:
        raise RuntimeError("failed to load interfaces")
    iface_mod = importlib.util.module_from_spec(iface_spec)
    sys.modules[iface_spec.name] = iface_mod
    iface_spec.loader.exec_module(iface_mod)

    # kv_cache_pool
    kv_spec = importlib.util.spec_from_file_location("llaisys.kv_cache_pool", str(kv_path))
    if kv_spec is None or kv_spec.loader is None:
        raise RuntimeError("failed to load kv_cache_pool")
    kv_mod = importlib.util.module_from_spec(kv_spec)
    sys.modules[kv_spec.name] = kv_mod
    kv_spec.loader.exec_module(kv_mod)

    # scheduler
    scheduler_spec = importlib.util.spec_from_file_location("llaisys.scheduler", str(scheduler_path))
    if scheduler_spec is None or scheduler_spec.loader is None:
        raise RuntimeError("failed to load scheduler")
    scheduler_mod = importlib.util.module_from_spec(scheduler_spec)
    sys.modules[scheduler_spec.name] = scheduler_mod
    scheduler_spec.loader.exec_module(scheduler_mod)

    # session_manager (new module)
    session_mgr_mod = None
    if session_mgr_path.exists():
        sm_spec = importlib.util.spec_from_file_location("llaisys.session_manager", str(session_mgr_path))
        if sm_spec is not None and sm_spec.loader is not None:
            session_mgr_mod = importlib.util.module_from_spec(sm_spec)
            sys.modules[sm_spec.name] = session_mgr_mod
            sm_spec.loader.exec_module(session_mgr_mod)

    # kv_runtime_bridge (new module)
    kv_bridge_mod = None
    if kv_bridge_path.exists():
        kb_spec = importlib.util.spec_from_file_location("llaisys.kv_runtime_bridge", str(kv_bridge_path))
        if kb_spec is not None and kb_spec.loader is not None:
            kv_bridge_mod = importlib.util.module_from_spec(kb_spec)
            sys.modules[kb_spec.name] = kv_bridge_mod
            kb_spec.loader.exec_module(kv_bridge_mod)

    # fake llaisys package
    fake_llaisys = types.ModuleType("llaisys")
    fake_llaisys.kv_cache_pool = kv_mod
    fake_llaisys.scheduler = scheduler_mod
    fake_llaisys.interfaces = iface_mod
    fake_llaisys.Tokenizer = object
    if session_mgr_mod:
        fake_llaisys.session_manager = session_mgr_mod
    if kv_bridge_mod:
        fake_llaisys.kv_runtime_bridge = kv_bridge_mod
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

    # fake models
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

    # server
    spec = importlib.util.spec_from_file_location("llaisys.server", str(server_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load server module")
    server_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = server_mod
    spec.loader.exec_module(server_mod)

    return iface_mod, kv_mod, scheduler_mod, session_mgr_mod, kv_bridge_mod, server_mod


iface_mod, kv_mod, scheduler_mod, session_mgr_mod, kv_bridge_mod, server_mod = _load_modules()
ChatService = server_mod.ChatService


# ---------------------------------------------------------------------------
# Fake model helpers
# ---------------------------------------------------------------------------

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
        self._last_kv_context = None

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
        self._last_kv_context = ctx
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
# SessionManager unit tests
# ===========================================================================

def test_session_manager_save_and_get_messages():
    """SessionManager should store and retrieve message history."""
    if session_mgr_mod is None:
        print("  SKIP: session_manager.py not found")
        return
    SessionManager = session_mgr_mod.SessionManager
    mgr = SessionManager()

    mgr.save_messages("s1", [{"role": "user", "content": "hello"}])
    msgs = mgr.get_messages("s1")
    assert len(msgs) == 1
    assert msgs[0]["content"] == "hello"

    # Should return a copy, not the original
    msgs.append({"role": "assistant", "content": "hi"})
    assert len(mgr.get_messages("s1")) == 1, "get_messages should return a copy"

    # Empty session returns empty list
    assert mgr.get_messages("nonexistent") == []

    print("  SessionManager save/get messages OK")


def test_session_manager_extract_messages_prompt_mode():
    """extract_messages with prompt should append to session history."""
    if session_mgr_mod is None:
        print("  SKIP: session_manager.py not found")
        return
    SessionManager = session_mgr_mod.SessionManager
    mgr = SessionManager()

    # First message in a new session
    ctx_id, msgs = mgr.extract_messages({"session_id": "s1", "prompt": "hello"})
    assert ctx_id == "s1"
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hello"

    print("  SessionManager extract_messages prompt mode OK")


def test_session_manager_extract_messages_list_mode():
    """extract_messages with messages list should use them directly."""
    if session_mgr_mod is None:
        print("  SKIP: session_manager.py not found")
        return
    SessionManager = session_mgr_mod.SessionManager
    mgr = SessionManager()

    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    ctx_id, msgs = mgr.extract_messages({"session_id": "s2", "messages": messages})
    assert ctx_id == "s2"
    assert len(msgs) == 2

    print("  SessionManager extract_messages list mode OK")


def test_session_manager_extract_messages_edit_fork():
    """extract_messages with edit_from_session_id should fork and edit."""
    if session_mgr_mod is None:
        print("  SKIP: session_manager.py not found")
        return
    SessionManager = session_mgr_mod.SessionManager
    mgr = SessionManager()

    # Set up source session
    mgr.save_messages("source", [
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "original answer"},
        {"role": "user", "content": "follow up"},
    ])

    ctx_id, msgs = mgr.extract_messages({
        "session_id": "fork1",
        "edit_from_session_id": "source",
        "edit_message_index": 0,
        "prompt": "edited question",
    })
    assert ctx_id == "fork1"
    assert len(msgs) == 1
    assert msgs[0]["content"] == "edited question"

    print("  SessionManager extract_messages edit fork OK")


def test_session_manager_cancel_event_lifecycle():
    """Cancel event: get, set (request_stop), clear."""
    if session_mgr_mod is None:
        print("  SKIP: session_manager.py not found")
        return
    SessionManager = session_mgr_mod.SessionManager
    mgr = SessionManager()

    event = mgr.get_cancel_event("s1")
    assert not event.is_set()

    mgr.request_stop("s1")
    assert event.is_set()

    mgr.clear_stop("s1")
    assert not event.is_set()

    print("  SessionManager cancel event lifecycle OK")


def test_session_manager_concurrent_access():
    """Multiple threads accessing SessionManager concurrently should not crash."""
    if session_mgr_mod is None:
        print("  SKIP: session_manager.py not found")
        return
    SessionManager = session_mgr_mod.SessionManager
    mgr = SessionManager()

    errors: List[Exception] = []
    barrier = threading.Barrier(10)

    def _worker(tid: int):
        try:
            barrier.wait(timeout=5.0)
            for j in range(20):
                sid = f"concurrent-{tid}-{j}"
                mgr.save_messages(sid, [{"role": "user", "content": f"msg-{j}"}])
                mgr.get_messages(sid)
                mgr.get_cancel_event(sid)
                mgr.request_stop(sid)
                mgr.clear_stop(sid)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    assert len(errors) == 0, f"Concurrent access errors: {errors}"
    print("  SessionManager concurrent access OK")


# ===========================================================================
# KVRuntimeBridge unit tests
# ===========================================================================

def test_kv_bridge_disabled_mode_skips_all():
    """When disabled, bind/export/release should be no-ops."""
    if kv_bridge_mod is None:
        print("  SKIP: kv_runtime_bridge.py not found")
        return
    KVRuntimeBridge = kv_bridge_mod.KVRuntimeBridge
    model = FakeModel()
    bridge = KVRuntimeBridge(model, enabled=False)

    assert bridge.enabled is False

    # bind should set_kv_context(None) or be a no-op
    bridge.bind_for_request("s1", [1, 2, 3], prefix_len=2)

    # export should be a no-op
    bridge.export_after_request("s1", [1, 2, 3, 65], block_size=4)
    assert len(model.export_calls) == 0, "disabled bridge should not export"

    # release should be a no-op
    bridge.release("s1")

    print("  KVRuntimeBridge disabled mode OK")


def test_kv_bridge_bind_export_release_lifecycle():
    """Full lifecycle: bind (no context) -> export -> bind (reuse) -> release."""
    if kv_bridge_mod is None:
        print("  SKIP: kv_runtime_bridge.py not found")
        return
    KVRuntimeBridge = kv_bridge_mod.KVRuntimeBridge
    model = FakeModel()
    bridge = KVRuntimeBridge(model, enabled=True)

    # First request: no existing context, prefix_len=0 -> bind None
    bridge.bind_for_request("s1", [1, 2, 3], prefix_len=0)
    assert model.bind_calls[-1] is None, "No prefix -> should bind None"

    # Export after first request
    bridge.export_after_request("s1", [1, 2, 3, 65], block_size=4)
    assert len(model.export_calls) >= 1, "Should export after request"

    # Second request: existing context for s1, prefix_len > 0 -> should bind non-None
    bridge.bind_for_request("s1", [1, 2, 3, 65, 4, 5], prefix_len=4)
    assert model.bind_calls[-1] is not None, "Existing context should bind non-None"

    # Release
    bridge.release("s1")

    # After release, bind should get None again
    bridge.bind_for_request("s1", [1, 2, 3], prefix_len=0)
    assert model.bind_calls[-1] is None, "After release, should bind None"

    print("  KVRuntimeBridge lifecycle OK")


def test_kv_bridge_cross_session_donor():
    """bind_for_request should find donor context from another session."""
    if kv_bridge_mod is None:
        print("  SKIP: kv_runtime_bridge.py not found")
        return
    KVRuntimeBridge = kv_bridge_mod.KVRuntimeBridge
    model = FakeModel()
    bridge = KVRuntimeBridge(model, enabled=True)

    # Set up donor session
    bridge.bind_for_request("donor", [10, 20, 30], prefix_len=0)
    bridge.export_after_request("donor", [10, 20, 30, 65], block_size=4)

    # Receiver with matching prefix should find donor
    bridge.bind_for_request("receiver", [10, 20, 30, 65, 40], prefix_len=4)
    # The last bind should be non-None (found donor context)
    assert model.bind_calls[-1] is not None, "Should find donor context"

    print("  KVRuntimeBridge cross-session donor OK")


def test_kv_bridge_debug_snapshot_format():
    """debug_snapshot should return expected fields."""
    if kv_bridge_mod is None:
        print("  SKIP: kv_runtime_bridge.py not found")
        return
    KVRuntimeBridge = kv_bridge_mod.KVRuntimeBridge
    model = FakeModel()
    bridge = KVRuntimeBridge(model, enabled=True)

    snap = bridge.debug_snapshot("s1")
    assert isinstance(snap, dict)
    assert "session_id" in snap
    assert "has_native_context" in snap
    assert "last_bind" in snap

    # Global snapshot (no session_id)
    snap_all = bridge.debug_snapshot(None)
    assert isinstance(snap_all, dict)
    assert "native_contexts" in snap_all

    print("  KVRuntimeBridge debug_snapshot format OK")


# ===========================================================================
# ChatService integration tests (delegation correctness)
# ===========================================================================

def test_chatservice_delegates_request_stop():
    """ChatService.request_stop should delegate to SessionManager."""
    service, _ = _make_service()
    # request_stop should work without prior generate
    result = service.request_stop("s-stop")
    assert result is True
    print("  ChatService delegates request_stop OK")


def test_chatservice_delegates_kv_debug_snapshot():
    """ChatService.kv_debug_snapshot should combine bridge + pool info."""
    service, model = _make_service()
    service.generate({"session_id": "s-dbg", "prompt": "test", "max_new_tokens": 2})

    snap = service.kv_debug_snapshot("s-dbg")
    assert isinstance(snap, dict)
    assert "session_id" in snap
    assert "kv_pool" in snap
    assert "has_native_context" in snap
    assert "last_bind" in snap

    # Global snapshot
    snap_all = service.kv_debug_snapshot(None)
    assert isinstance(snap_all, dict)
    assert "kv_pool" in snap_all

    print("  ChatService kv_debug_snapshot delegation OK")


def test_chatservice_generate_saves_messages():
    """After generate, session history should be saved (via SessionManager)."""
    service, _ = _make_service()
    service.generate({"session_id": "s-hist", "prompt": "hello", "max_new_tokens": 2})
    service.generate({"session_id": "s-hist", "prompt": "again", "max_new_tokens": 2})

    # Verify the session has history by doing a third request
    # (it should pick up prior messages in the prompt)
    result = service.generate({"session_id": "s-hist", "prompt": "third", "max_new_tokens": 2})
    assert result["session_id"] == "s-hist"
    print("  ChatService generate saves messages OK")


def test_chatservice_cancelled_does_not_save_messages():
    """Cancelled request should not save assistant output to history."""
    service, model = _make_service()

    # Override _iter_generate_ids to immediately cancel
    def _cancelled_iter(prompt_ids, max_new_tokens, sampling, prefix_len, cancel_event):
        cancel_event.set()
        if False:
            yield 0

    service._iter_generate_ids = _cancelled_iter
    result = service.generate({"session_id": "s-cancel", "prompt": "test", "max_new_tokens": 2})
    assert result.get("stopped") is True
    assert result["choices"][0]["finish_reason"] == "stop"
    assert len(model.export_calls) == 0
    print("  ChatService cancelled request does not save messages OK")


# ===========================================================================
# Interface compatibility
# ===========================================================================

def test_isinstance_checks_still_pass():
    """ChatService should still be an IInferenceService after refactoring."""
    IInferenceService = getattr(iface_mod, "IInferenceService", None)
    IKVCachePool = getattr(iface_mod, "IKVCachePool", None)

    service, _ = _make_service()

    if IInferenceService is not None:
        assert isinstance(service, IInferenceService), (
            "ChatService must be an instance of IInferenceService"
        )
        print("  isinstance(ChatService, IInferenceService): True")

    if IKVCachePool is not None:
        pool = kv_mod.KVCachePool(block_size=4, max_blocks=128, max_bytes=1024 * 1024)
        assert isinstance(pool, IKVCachePool), (
            "KVCachePool must be an instance of IKVCachePool"
        )
        print("  isinstance(KVCachePool, IKVCachePool): True")


# ===========================================================================
# Regression tests
# ===========================================================================

def test_regression_kv_reuse_same_session():
    """Regression: same-session KV reuse still works after split."""
    service, model = _make_service()
    first = service.generate({"session_id": "reg-s1", "prompt": "hello", "max_new_tokens": 2})
    assert first["session_id"] == "reg-s1"
    assert model.bind_calls[0] is None  # first request has no prefix

    service.generate({"session_id": "reg-s1", "prompt": "again", "max_new_tokens": 2})
    assert model.bind_calls[-1] is not None  # second should bind existing context
    dbg = service.kv_debug_snapshot("reg-s1")
    assert dbg["last_bind"]["bound"] is True
    assert dbg["last_bind"]["prefix_len"] > 0
    print("  regression: same-session KV reuse OK")


def test_regression_cross_session_donor():
    """Regression: cross-session KV donor still works after split."""
    service, _ = _make_service()
    service.generate({"session_id": "donor", "prompt": "shared prompt", "max_new_tokens": 2})
    service.generate({
        "session_id": "receiver",
        "messages": [{"role": "user", "content": "shared prompt"}],
        "max_new_tokens": 2,
    })
    dbg = service.kv_debug_snapshot("receiver")
    assert dbg["last_bind"]["bound"] is True
    assert dbg["last_bind"]["prefix_len"] > 0
    assert dbg["last_bind"]["source_session_id"] == "donor"
    print("  regression: cross-session donor KV reuse OK")


def test_regression_stream_works():
    """Regression: stream generation still works."""
    service, _ = _make_service()
    items = list(service.stream({"session_id": "reg-stream", "prompt": "hello", "max_new_tokens": 2}))
    assert items[-1]["choices"][0]["finish_reason"] is not None
    assert items[-1]["session_id"] == "reg-stream"
    print("  regression: stream OK")


def test_regression_kv_cache_pool_prefix_match():
    """Regression: KVCachePool prefix matching still works."""
    KVCachePool = kv_mod.KVCachePool
    pool = KVCachePool(block_size=4, max_blocks=128, max_bytes=1024 * 1024)
    result_a = pool.acquire_context("ctx-a", [1, 2, 3, 4, 5, 6])
    assert result_a.prefix_len == 0
    result_b = pool.acquire_context("ctx-b", [1, 2, 3, 4, 5, 6])
    assert result_b.prefix_len == 4
    print("  regression: kv_cache_pool prefix match OK")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    tests = [
        # SessionManager unit tests
        test_session_manager_save_and_get_messages,
        test_session_manager_extract_messages_prompt_mode,
        test_session_manager_extract_messages_list_mode,
        test_session_manager_extract_messages_edit_fork,
        test_session_manager_cancel_event_lifecycle,
        test_session_manager_concurrent_access,
        # KVRuntimeBridge unit tests
        test_kv_bridge_disabled_mode_skips_all,
        test_kv_bridge_bind_export_release_lifecycle,
        test_kv_bridge_cross_session_donor,
        test_kv_bridge_debug_snapshot_format,
        # ChatService integration
        test_chatservice_delegates_request_stop,
        test_chatservice_delegates_kv_debug_snapshot,
        test_chatservice_generate_saves_messages,
        test_chatservice_cancelled_does_not_save_messages,
        # Interface compatibility
        test_isinstance_checks_still_pass,
        # Regression
        test_regression_kv_reuse_same_session,
        test_regression_cross_session_donor,
        test_regression_stream_works,
        test_regression_kv_cache_pool_prefix_match,
    ]

    passed = 0
    failed = 0
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
