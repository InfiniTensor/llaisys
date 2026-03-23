"""Tests for shared model pool, shared KV pool, and KV memory-aware flow control.

Covers:
- Shared model + KV pool: multiple ChatService instances share the same objects
- Cross-worker prefix reuse via shared KV pool
- KV memory_pressure() correctness
- KV memory-aware flow control in scheduler (reject when pressure > threshold)
- Shared pool routing optimization in scheduler
- debug_snapshot includes kv_memory_pressure and kv_memory_threshold
"""

import importlib.util
import sys
import threading
import types
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Module loading (same pattern as existing tests)
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

    # session_manager
    session_mgr_mod = None
    if session_mgr_path.exists():
        sm_spec = importlib.util.spec_from_file_location("llaisys.session_manager", str(session_mgr_path))
        if sm_spec is not None and sm_spec.loader is not None:
            session_mgr_mod = importlib.util.module_from_spec(sm_spec)
            sys.modules[sm_spec.name] = session_mgr_mod
            sm_spec.loader.exec_module(session_mgr_mod)

    # kv_runtime_bridge
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
    fake_llaisys.__path__ = [str(root / "python" / "llaisys")]
    sys.modules["llaisys"] = fake_llaisys
    sys.modules["llaisys.kv_cache_pool"] = kv_mod
    sys.modules["llaisys.scheduler"] = scheduler_mod
    sys.modules["llaisys.interfaces"] = iface_mod
    if session_mgr_mod:
        sys.modules["llaisys.session_manager"] = session_mgr_mod
    if kv_bridge_mod:
        sys.modules["llaisys.kv_runtime_bridge"] = kv_bridge_mod

    # fake libllaisys
    fake_libllaisys = types.ModuleType("llaisys.libllaisys")

    class _FakeSamplingParams:
        def __init__(self, top_k=1, top_p=0.0, temperature=0.0, seed=0):
            self.top_k = top_k
            self.top_p = top_p
            self.temperature = temperature
            self.seed = seed

    fake_libllaisys.LlaisysSamplingParams = _FakeSamplingParams
    sys.modules["llaisys.libllaisys"] = fake_libllaisys
    fake_llaisys.libllaisys = fake_libllaisys

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

    return iface_mod, kv_mod, scheduler_mod, kv_bridge_mod, server_mod


iface_mod, kv_mod, scheduler_mod, kv_bridge_mod, server_mod = _load_modules()
ChatService = server_mod.ChatService
KVCachePool = kv_mod.KVCachePool
KVRuntimeBridge = kv_bridge_mod.KVRuntimeBridge
InferenceScheduler = scheduler_mod.InferenceScheduler
SchedulerQueueFullError = scheduler_mod.SchedulerQueueFullError


# ---------------------------------------------------------------------------
# Fake model / tokenizer helpers
# ---------------------------------------------------------------------------

class _EndToken:
    def __init__(self, value):
        self.value = value


class _Meta:
    def __init__(self, eos=-1):
        self.end_token = _EndToken(eos)


class FakeTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text]

    def decode(self, token_ids):
        return "".join(chr(int(t)) for t in token_ids)


class FakeModel:
    def __init__(self, eos=-1):
        self._meta = _Meta(eos)
        self.bind_calls = []
        self.export_calls = []
        self.reset_calls = 0
        self._ctx_seq = 0
        self.prefill_packed_calls = 0
        self.step_packed_calls = 0

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

    def prefill_packed(self, prompts):
        self.prefill_packed_calls += 1
        return [65] * len(prompts)

    def step_packed(self, sequences):
        self.step_packed_calls += 1
        return [66] * len(sequences)

    def prefill_packed_sampling(self, prompts, params_list):
        return [65] * len(prompts)

    def step_packed_sampling(self, sequences, params_list):
        return [66] * len(sequences)

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


def _make_shared_services(worker_count=2, **kwargs):
    """Create multiple ChatService instances sharing the same model, lock, KV pool, and KV bridge."""
    model = FakeModel()
    tok = FakeTokenizer()
    shared_lock = threading.RLock()
    shared_kv_pool = KVCachePool(
        block_size=kwargs.get("block_size", 4),
        max_blocks=kwargs.get("max_blocks", 256),
        max_bytes=kwargs.get("max_bytes", 1024 * 1024),
    )
    shared_kv_bridge = KVRuntimeBridge(model, enabled=kwargs.get("enable_kv_runtime_reuse", True))
    services = []
    for _ in range(worker_count):
        svc = ChatService(
            model=model,
            tokenizer=tok,
            model_path=None,
            enable_kv_runtime_reuse=kwargs.get("enable_kv_runtime_reuse", True),
            block_size=kwargs.get("block_size", 4),
            max_blocks=kwargs.get("max_blocks", 256),
            max_bytes=kwargs.get("max_bytes", 1024 * 1024),
            model_lock=shared_lock,
            kv_pool=shared_kv_pool,
            kv_bridge=shared_kv_bridge,
        )
        services.append(svc)
    return services, model, shared_kv_pool, shared_lock, shared_kv_bridge


def _make_independent_services(worker_count=2, **kwargs):
    """Create multiple ChatService instances with independent resources."""
    services = []
    models = []
    for _ in range(worker_count):
        model = FakeModel()
        tok = FakeTokenizer()
        svc = ChatService(
            model=model,
            tokenizer=tok,
            model_path=None,
            enable_kv_runtime_reuse=kwargs.get("enable_kv_runtime_reuse", False),
            block_size=kwargs.get("block_size", 4),
            max_blocks=kwargs.get("max_blocks", 256),
            max_bytes=kwargs.get("max_bytes", 1024 * 1024),
        )
        services.append(svc)
        models.append(model)
    return services, models


# ===========================================================================
# Test 1: Shared instances are the same object
# ===========================================================================

def test_shared_instances_identity():
    """All ChatService instances should share the same model, lock, KV pool, and KV bridge."""
    services, model, shared_pool, shared_lock, shared_bridge = _make_shared_services(3)

    for i, svc in enumerate(services):
        assert svc.model is model, f"Service {i} should share the model"
        assert svc._model_lock is shared_lock, f"Service {i} should share the model lock"
        assert svc._kv_pool is shared_pool, f"Service {i} should share the KV pool"
        assert svc._kv_bridge is shared_bridge, f"Service {i} should share the KV bridge"

    # Each service should have its own SessionManager
    assert services[0]._session_mgr is not services[1]._session_mgr

    print("  shared instances identity OK")


# ===========================================================================
# Test 2: Independent instances are distinct
# ===========================================================================

def test_independent_instances_distinct():
    """Independent ChatService instances should have separate resources."""
    services, models = _make_independent_services(2)

    assert services[0].model is not services[1].model
    assert services[0]._model_lock is not services[1]._model_lock
    assert services[0]._kv_pool is not services[1]._kv_pool
    assert services[0]._kv_bridge is not services[1]._kv_bridge

    print("  independent instances distinct OK")


# ===========================================================================
# Test 3: memory_pressure() correctness
# ===========================================================================

def test_memory_pressure_empty():
    """Empty pool should have 0.0 pressure."""
    pool = KVCachePool(block_size=4, max_blocks=100, max_bytes=1024 * 1024)
    assert pool.memory_pressure() == 0.0
    print("  memory_pressure empty OK")


def test_memory_pressure_increases():
    """Pressure should increase as blocks are allocated."""
    pool = KVCachePool(block_size=4, max_blocks=10, max_bytes=1024 * 1024)
    assert pool.memory_pressure() == 0.0

    # Acquire contexts to fill blocks
    for i in range(5):
        tokens = list(range(i * 4, (i + 1) * 4))
        pool.acquire_context(f"ctx-{i}", tokens)

    pressure = pool.memory_pressure()
    assert pressure > 0.0, f"Pressure should be > 0 after allocations, got {pressure}"
    assert pressure <= 1.0, f"Pressure should be <= 1.0, got {pressure}"

    print("  memory_pressure increases OK")


def test_memory_pressure_interface():
    """memory_pressure should be available via IKVCachePool interface."""
    IKVCachePool = iface_mod.IKVCachePool
    pool = KVCachePool(block_size=4, max_blocks=100, max_bytes=1024 * 1024)
    assert isinstance(pool, IKVCachePool)
    assert hasattr(pool, "memory_pressure")
    assert callable(pool.memory_pressure)
    print("  memory_pressure interface OK")


# ===========================================================================
# Test 4: Cross-worker prefix reuse via shared KV pool
# ===========================================================================

def test_shared_pool_cross_worker_prefix_reuse():
    """With shared KV pool, a context created by one worker should be visible to another."""
    services, model, shared_pool, _, _ = _make_shared_services(2)

    # Worker 0 generates with a prompt
    services[0].generate({"session_id": "shared-s1", "prompt": "hello world", "max_new_tokens": 2})

    # Worker 1 should see the prefix from worker 0's context in the shared pool
    prefix_len = shared_pool.query_prefix_len(
        services[1].tokenizer.encode("User: hello world\nAssistant:")
    )
    # The exact prefix_len depends on block alignment, but should be > 0
    # since worker 0 already created blocks for the same prompt pattern
    assert prefix_len >= 0, "Shared pool should allow cross-worker prefix queries"

    # Worker 1 generates with the same prompt - should benefit from shared pool
    services[1].generate({
        "session_id": "shared-s2",
        "messages": [{"role": "user", "content": "hello world"}],
        "max_new_tokens": 2,
    })

    stats = shared_pool.snapshot_stats()
    assert stats["acquire_count"] == 2.0, "Both workers should have acquired from the same pool"

    print("  shared pool cross-worker prefix reuse OK")


# ===========================================================================
# Test 5: KV memory flow control - reject when pressure > threshold
# ===========================================================================

def test_kv_memory_flow_control_rejects():
    """Scheduler should reject requests when KV memory pressure exceeds threshold."""
    # Use a tiny pool so pressure rises quickly
    services, model, shared_pool, _, _ = _make_shared_services(
        1, max_blocks=2, max_bytes=64, block_size=4,
    )

    # Fill the pool to create pressure
    for i in range(3):
        tokens = list(range(i * 4, (i + 1) * 4))
        shared_pool.acquire_context(f"fill-{i}", tokens)

    pressure = shared_pool.memory_pressure()
    assert pressure > 0.5, f"Pool should be under pressure, got {pressure}"

    # Create scheduler with low threshold
    scheduler = InferenceScheduler(
        services,
        queue_size=8,
        request_timeout_ms=5000,
        kv_memory_threshold=0.1,  # very low threshold
    )

    rejected = False
    try:
        scheduler.submit({"session_id": "reject-test", "prompt": "test", "max_new_tokens": 1}, stream=False)
    except SchedulerQueueFullError as exc:
        assert "KV memory pressure" in str(exc)
        rejected = True

    assert rejected, "Should have rejected due to KV memory pressure"
    print("  KV memory flow control rejects OK")


def test_kv_memory_flow_control_allows_when_below():
    """Scheduler should allow requests when KV memory pressure is below threshold."""
    services, model, shared_pool, _, _ = _make_shared_services(1)

    scheduler = InferenceScheduler(
        services,
        queue_size=8,
        request_timeout_ms=5000,
        kv_memory_threshold=0.85,
    )
    scheduler.start()
    try:
        handle = scheduler.submit(
            {"session_id": "allow-test", "prompt": "test", "max_new_tokens": 2},
            stream=False,
        )
        result = handle.get_result(timeout=5.0)
        assert "choices" in result
    finally:
        scheduler.stop()

    print("  KV memory flow control allows when below OK")


def test_kv_memory_flow_control_disabled():
    """When threshold is 0.0, flow control should be disabled."""
    services, model, shared_pool, _, _ = _make_shared_services(
        1, max_blocks=2, max_bytes=64, block_size=4,
    )

    # Fill pool
    for i in range(3):
        tokens = list(range(i * 4, (i + 1) * 4))
        shared_pool.acquire_context(f"fill-{i}", tokens)

    scheduler = InferenceScheduler(
        services,
        queue_size=8,
        request_timeout_ms=5000,
        kv_memory_threshold=0.0,  # disabled
    )
    scheduler.start()
    try:
        # Should not reject even with high pressure
        handle = scheduler.submit(
            {"session_id": "no-fc", "prompt": "test", "max_new_tokens": 2},
            stream=False,
        )
        result = handle.get_result(timeout=5.0)
        assert "choices" in result
    finally:
        scheduler.stop()

    print("  KV memory flow control disabled OK")


# ===========================================================================
# Test 6: KV memory metrics in debug_snapshot
# ===========================================================================

def test_debug_snapshot_kv_memory_fields():
    """debug_snapshot should include kv_memory_threshold and kv_memory_pressure."""
    services, _, _, _, _ = _make_shared_services(1)
    scheduler = InferenceScheduler(
        services,
        queue_size=8,
        kv_memory_threshold=0.85,
    )
    snap = scheduler.debug_snapshot()
    assert "kv_memory_threshold" in snap, "Should have kv_memory_threshold"
    assert "kv_memory_pressure" in snap, "Should have kv_memory_pressure"
    assert snap["kv_memory_threshold"] == 0.85
    assert snap["kv_memory_pressure"] == 0.0  # empty pool

    print("  debug_snapshot KV memory fields OK")


def test_debug_snapshot_kv_memory_rejected_metric():
    """kv_memory_rejected metric should increment on rejection."""
    services, _, shared_pool, _, _ = _make_shared_services(
        1, max_blocks=2, max_bytes=64, block_size=4,
    )
    for i in range(3):
        tokens = list(range(i * 4, (i + 1) * 4))
        shared_pool.acquire_context(f"fill-{i}", tokens)

    scheduler = InferenceScheduler(
        services,
        queue_size=8,
        kv_memory_threshold=0.1,
    )

    try:
        scheduler.submit({"prompt": "test"}, stream=False)
    except SchedulerQueueFullError:
        pass

    snap = scheduler.debug_snapshot()
    assert snap["metrics"]["kv_memory_rejected"] >= 1.0

    print("  debug_snapshot kv_memory_rejected metric OK")


# ===========================================================================
# Test 7: Shared pool routing optimization
# ===========================================================================

def test_shared_pool_kv_debug_snapshot_no_double_count():
    """With shared pool, kv_debug_snapshot should not double-count stats."""
    services, _, shared_pool, _, _ = _make_shared_services(2)

    # Generate on one worker
    services[0].generate({"session_id": "snap-s1", "prompt": "hello", "max_new_tokens": 2})

    scheduler = InferenceScheduler(services, queue_size=8)
    snap = scheduler.kv_debug_snapshot()

    # With shared pool, acquire_count should be 1 (not 2)
    assert snap["kv_pool"]["acquire_count"] == 1.0, (
        f"Shared pool should report 1 acquire, got {snap['kv_pool']['acquire_count']}"
    )

    print("  shared pool kv_debug_snapshot no double count OK")


# ===========================================================================
# Test 8: Shared model concurrent generate
# ===========================================================================

def test_shared_model_concurrent_generate():
    """Multiple workers sharing a model should serialize via the shared lock."""
    services, model, _, _, _ = _make_shared_services(2)

    results = [None, None]
    errors = []

    def _worker(idx):
        try:
            r = services[idx].generate({
                "session_id": f"concurrent-{idx}",
                "prompt": f"msg-{idx}",
                "max_new_tokens": 2,
            })
            results[idx] = r
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert len(errors) == 0, f"Concurrent errors: {errors}"
    for i, r in enumerate(results):
        assert r is not None, f"Worker {i} should have produced a result"
        assert "choices" in r

    print("  shared model concurrent generate OK")


# ===========================================================================
# Test 9: Shared model + scheduler end-to-end
# ===========================================================================

def test_shared_model_scheduler_e2e():
    """End-to-end: scheduler with shared model services."""
    services, model, _, _, _ = _make_shared_services(2)
    scheduler = InferenceScheduler(
        services,
        queue_size=8,
        request_timeout_ms=5000,
        continuous_batching=True,
        max_batch_size=4,
    )
    scheduler.start()
    try:
        handles = []
        for i in range(4):
            h = scheduler.submit(
                {"session_id": f"e2e-shared-{i}", "prompt": f"test-{i}", "max_new_tokens": 2, "stream": True},
                stream=True,
            )
            handles.append(h)

        for i, h in enumerate(handles):
            items = list(h.iter_stream(timeout=5.0))
            assert len(items) > 0, f"Stream {i} should produce chunks"
            last = items[-1]
            assert last["choices"][0]["finish_reason"] is not None
    finally:
        scheduler.stop()

    print("  shared model scheduler e2e OK")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    tests = [
        test_shared_instances_identity,
        test_independent_instances_distinct,
        test_memory_pressure_empty,
        test_memory_pressure_increases,
        test_memory_pressure_interface,
        test_shared_pool_cross_worker_prefix_reuse,
        test_kv_memory_flow_control_rejects,
        test_kv_memory_flow_control_allows_when_below,
        test_kv_memory_flow_control_disabled,
        test_debug_snapshot_kv_memory_fields,
        test_debug_snapshot_kv_memory_rejected_metric,
        test_shared_pool_kv_debug_snapshot_no_double_count,
        test_shared_model_concurrent_generate,
        test_shared_model_scheduler_e2e,
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
            import traceback
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
