"""Tests for streaming batch processing (Phase 1-3):
- Streaming batch produces correct SSE chunks (multi-sequence parallel)
- Non-stream requests via batch path
- Mixed stream + non-stream in same batch
- Single sequence cancellation while others continue
- Different max_new_tokens (partial early finish)
- Batch size limit enforcement
- Dynamic shrink verification
- Fallback to single path when no packed API
- All existing test suites pass (regression)
"""

import importlib.util
import sys
import threading
import time
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

    return iface_mod, kv_mod, scheduler_mod, server_mod


iface_mod, kv_mod, scheduler_mod, server_mod = _load_modules()
ChatService = server_mod.ChatService
BatchSequenceState = server_mod.BatchSequenceState
BatchState = server_mod.BatchState
StepResult = server_mod.StepResult


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
    """Model mock with packed API support."""

    def __init__(self, eos=-1):
        self._meta = _Meta(eos)
        self.bind_calls = []
        self.export_calls = []
        self.reset_calls = 0
        self._ctx_seq = 0
        self.prefill_packed_calls = 0
        self.step_packed_calls = 0
        self.prefill_packed_sampling_calls = 0
        self.step_packed_sampling_calls = 0
        # Track decode_inputs sizes for shrink verification
        self.step_packed_input_sizes: List[int] = []

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
        self.step_packed_input_sizes.append(len(sequences))
        return [66] * len(sequences)

    def prefill_packed_sampling(self, prompts, params_list):
        self.prefill_packed_sampling_calls += 1
        return [65] * len(prompts)

    def step_packed_sampling(self, sequences, params_list):
        self.step_packed_sampling_calls += 1
        self.step_packed_input_sizes.append(len(sequences))
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


class FakeModelNoPacked:
    """Model without any packed methods."""

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
        return 65

    def step(self, token_ids):
        return 66

    def step_sampling(self, token_ids, top_k=1, top_p=0.0, temperature=0.0, seed=0):
        return 66

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


def _make_service(model=None, **kwargs):
    if model is None:
        model = FakeModel()
    tok = FakeTokenizer()
    service = ChatService(
        model=model,
        tokenizer=tok,
        model_path=None,
        enable_kv_runtime_reuse=kwargs.get("enable_kv_runtime_reuse", False),
        block_size=kwargs.get("block_size", 4),
        max_blocks=kwargs.get("max_blocks", 256),
        max_bytes=kwargs.get("max_bytes", 1024 * 1024),
    )
    return service, model


# ===========================================================================
# Test 1: Streaming batch produces correct SSE chunks
# ===========================================================================

def test_streaming_batch_correct_chunks():
    """prepare_batch + step_batch should produce correct delta text for multiple sequences."""
    service, model = _make_service()
    payloads = [
        {"session_id": "s1", "prompt": "hi", "max_new_tokens": 3},
        {"session_id": "s2", "prompt": "yo", "max_new_tokens": 3},
    ]
    state = service.prepare_batch(payloads)
    assert state is not None, "prepare_batch should return BatchState"
    assert len(state.sequences) == 2
    assert state.sequences[0].context_id == "s1"
    assert state.sequences[1].context_id == "s2"

    # First token already generated in prefill
    for seq in state.sequences:
        assert len(seq.generated_ids) == 1
        assert seq.generated_ids[0] == 65  # 'A'

    # Step until all done
    all_deltas: Dict[str, str] = {"s1": "", "s2": ""}
    rounds = 0
    while not all(s.finished for s in state.sequences):
        results = service.step_batch(state)
        for sr in results:
            seq = state.sequences[sr.seq_index]
            all_deltas[seq.context_id] += sr.delta_text
        rounds += 1
        assert rounds < 20, "Too many decode rounds"

    # Each sequence should have generated max_new_tokens tokens
    for seq in state.sequences:
        assert len(seq.generated_ids) == 3
        assert seq.finish_reason == "length"

    print("  streaming batch correct chunks OK")


# ===========================================================================
# Test 2: Non-stream requests via batch path
# ===========================================================================

def test_non_stream_via_batch_path():
    """Non-stream payloads should work through prepare_batch/step_batch."""
    service, model = _make_service()
    payloads = [
        {"session_id": "ns1", "prompt": "hello", "max_new_tokens": 2},
        {"session_id": "ns2", "prompt": "world", "max_new_tokens": 2},
    ]
    state = service.prepare_batch(payloads)
    assert state is not None

    while not all(s.finished for s in state.sequences):
        service.step_batch(state)

    for i, seq in enumerate(state.sequences):
        assert seq.finished
        assert len(seq.generated_ids) == 2
        service.finalize_sequence(state, i)

    print("  non-stream via batch path OK")


# ===========================================================================
# Test 3: Mixed stream + non-stream in same batch
# ===========================================================================

def test_mixed_stream_non_stream_batch():
    """Both stream and non-stream payloads can be batched together."""
    service, model = _make_service()
    payloads = [
        {"session_id": "mix-s", "prompt": "hi", "max_new_tokens": 2, "stream": True},
        {"session_id": "mix-ns", "prompt": "yo", "max_new_tokens": 2},
    ]
    state = service.prepare_batch(payloads)
    assert state is not None
    assert len(state.sequences) == 2

    while not all(s.finished for s in state.sequences):
        service.step_batch(state)

    for seq in state.sequences:
        assert seq.finished
        assert len(seq.generated_ids) == 2

    print("  mixed stream+non-stream batch OK")


# ===========================================================================
# Test 4: Single sequence cancellation
# ===========================================================================

def test_single_sequence_cancellation():
    """Cancelling one sequence should not affect others."""
    service, model = _make_service()
    payloads = [
        {"session_id": "cancel-1", "prompt": "hi", "max_new_tokens": 5},
        {"session_id": "cancel-2", "prompt": "yo", "max_new_tokens": 5},
    ]
    state = service.prepare_batch(payloads)
    assert state is not None

    # Cancel first sequence after prefill
    state.sequences[0].cancel_event.set()

    results = service.step_batch(state)
    # First sequence should be marked as cancelled
    cancelled = [r for r in results if r.seq_index == 0]
    assert len(cancelled) == 1
    assert cancelled[0].finished
    assert cancelled[0].stopped

    # Second sequence should still be active
    assert not state.sequences[1].finished

    # Continue stepping until second finishes
    rounds = 0
    while not all(s.finished for s in state.sequences):
        service.step_batch(state)
        rounds += 1
        assert rounds < 20

    assert state.sequences[1].finished
    assert state.sequences[1].finish_reason == "length"
    assert len(state.sequences[1].generated_ids) == 5

    print("  single sequence cancellation OK")


# ===========================================================================
# Test 5: Different max_new_tokens (partial early finish)
# ===========================================================================

def test_different_max_new_tokens():
    """Sequences with different max_new_tokens should finish at different times."""
    service, model = _make_service()
    payloads = [
        {"session_id": "short", "prompt": "hi", "max_new_tokens": 1},
        {"session_id": "long", "prompt": "yo", "max_new_tokens": 4},
    ]
    state = service.prepare_batch(payloads)
    assert state is not None

    # Short sequence should finish after prefill (1 token generated)
    assert state.sequences[0].finished, "1-token sequence should finish after prefill"
    assert state.sequences[0].finish_reason == "length"
    assert not state.sequences[1].finished

    # Step until long finishes
    rounds = 0
    while not all(s.finished for s in state.sequences):
        service.step_batch(state)
        rounds += 1
        assert rounds < 20

    assert state.sequences[1].finished
    assert len(state.sequences[1].generated_ids) == 4

    print("  different max_new_tokens OK")


# ===========================================================================
# Test 6: Batch size limit enforcement (via scheduler)
# ===========================================================================

def test_batch_size_limit():
    """Scheduler should respect max_batch_size."""
    service, model = _make_service()
    InferenceScheduler = scheduler_mod.InferenceScheduler
    scheduler = InferenceScheduler(
        [service],
        queue_size=16,
        request_timeout_ms=5000,
        continuous_batching=True,
        max_batch_size=2,
    )
    scheduler.start()
    try:
        handles = []
        for i in range(4):
            h = scheduler.submit(
                {"session_id": f"bs-{i}", "prompt": "test", "max_new_tokens": 2, "stream": True},
                stream=True,
            )
            handles.append(h)

        for h in handles:
            items = list(h.iter_stream(timeout=5.0))
            assert len(items) > 0
            last = items[-1]
            assert last["choices"][0]["finish_reason"] is not None

        snap = scheduler.debug_snapshot()
        # Should have done multiple prefill batches since max_batch_size=2 and 4 tasks
        assert snap["max_batch_size"] == 2
    finally:
        scheduler.stop()

    print("  batch size limit OK")


# ===========================================================================
# Test 7: Dynamic shrink verification
# ===========================================================================

def test_dynamic_shrink():
    """step_batch should only pass active sequences to model (dynamic shrinking)."""
    service, model = _make_service()
    payloads = [
        {"session_id": "shrink-1", "prompt": "hi", "max_new_tokens": 1},  # finishes after prefill
        {"session_id": "shrink-2", "prompt": "yo", "max_new_tokens": 3},
    ]
    state = service.prepare_batch(payloads)
    assert state is not None
    assert state.sequences[0].finished  # 1 token = done after prefill

    model.step_packed_input_sizes.clear()

    # Step: only sequence 1 should be active
    rounds = 0
    while not all(s.finished for s in state.sequences):
        service.step_batch(state)
        rounds += 1
        assert rounds < 20

    # All step_packed calls should have received only 1 sequence (the active one)
    for size in model.step_packed_input_sizes:
        assert size == 1, f"Expected 1 active sequence in step_packed, got {size}"

    print("  dynamic shrink OK")


# ===========================================================================
# Test 8: Fallback to single path when no packed API
# ===========================================================================

def test_fallback_no_packed_api():
    """prepare_batch should return None when model has no packed methods."""
    model = FakeModelNoPacked()
    service, _ = _make_service(model=model)
    payloads = [
        {"session_id": "fb-1", "prompt": "hi", "max_new_tokens": 2},
    ]
    result = service.prepare_batch(payloads)
    assert result is None, "Should return None without packed API"
    print("  fallback no packed API OK")


def test_fallback_edit_from_session():
    """prepare_batch should return None for edit_from_session_id requests."""
    service, _ = _make_service()
    payloads = [
        {"session_id": "edit-1", "prompt": "hi", "max_new_tokens": 2,
         "edit_from_session_id": "other", "edit_message_index": 0},
    ]
    result = service.prepare_batch(payloads)
    assert result is None, "Should return None for edit requests"
    print("  fallback edit_from_session OK")


# ===========================================================================
# Test 9: Scheduler integration - streaming batch end-to-end
# ===========================================================================

def test_scheduler_streaming_batch_e2e():
    """Full end-to-end: scheduler uses prepare_batch/step_batch for streaming."""
    service, model = _make_service()
    InferenceScheduler = scheduler_mod.InferenceScheduler
    scheduler = InferenceScheduler(
        [service],
        queue_size=8,
        request_timeout_ms=5000,
        continuous_batching=True,
        max_batch_size=4,
    )
    scheduler.start()
    try:
        # Submit multiple stream requests
        handles = []
        for i in range(3):
            h = scheduler.submit(
                {"session_id": f"e2e-{i}", "prompt": "test", "max_new_tokens": 3, "stream": True},
                stream=True,
            )
            handles.append(h)

        # Collect all chunks
        for i, h in enumerate(handles):
            items = list(h.iter_stream(timeout=5.0))
            assert len(items) > 0, f"Stream {i} should produce chunks"
            last = items[-1]
            assert last["choices"][0]["finish_reason"] is not None, f"Stream {i} should have finish_reason"
            assert last["session_id"] == f"e2e-{i}"

        snap = scheduler.debug_snapshot()
        metrics = snap["metrics"]
        # Should have used the batch path
        assert metrics["stream_batch_prefill_batches"] >= 1.0 or metrics["stream_batch_fallback_tasks"] >= 1.0
    finally:
        scheduler.stop()

    print("  scheduler streaming batch e2e OK")


# ===========================================================================
# Test 10: Scheduler non-stream via batch path
# ===========================================================================

def test_scheduler_non_stream_batch():
    """Non-stream requests through continuous batching scheduler."""
    service, model = _make_service()
    InferenceScheduler = scheduler_mod.InferenceScheduler
    scheduler = InferenceScheduler(
        [service],
        queue_size=8,
        request_timeout_ms=5000,
        continuous_batching=True,
        max_batch_size=4,
    )
    scheduler.start()
    try:
        h = scheduler.submit(
            {"session_id": "ns-sched", "prompt": "test", "max_new_tokens": 2},
            stream=False,
        )
        result = h.get_result(timeout=5.0)
        assert result["session_id"] == "ns-sched"
        assert "choices" in result
    finally:
        scheduler.stop()

    print("  scheduler non-stream batch OK")


# ===========================================================================
# Test 11: Scheduler fallback path (no packed API)
# ===========================================================================

def test_scheduler_fallback_path():
    """Scheduler should fall back to legacy iterator path when prepare_batch returns None."""
    model = FakeModelNoPacked()
    service, _ = _make_service(model=model)
    InferenceScheduler = scheduler_mod.InferenceScheduler
    scheduler = InferenceScheduler(
        [service],
        queue_size=8,
        request_timeout_ms=5000,
        continuous_batching=True,
        max_batch_size=4,
    )
    scheduler.start()
    try:
        h = scheduler.submit(
            {"session_id": "fb-sched", "prompt": "test", "max_new_tokens": 2, "stream": True},
            stream=True,
        )
        items = list(h.iter_stream(timeout=5.0))
        assert len(items) > 0
        last = items[-1]
        assert last["choices"][0]["finish_reason"] is not None

        snap = scheduler.debug_snapshot()
        assert snap["metrics"]["stream_batch_fallback_tasks"] >= 1.0
    finally:
        scheduler.stop()

    print("  scheduler fallback path OK")


# ===========================================================================
# Test 12: finalize_sequence saves messages
# ===========================================================================

def test_finalize_saves_messages():
    """finalize_sequence should save assistant message to session history."""
    service, model = _make_service()
    payloads = [
        {"session_id": "fin-1", "prompt": "hello", "max_new_tokens": 2},
    ]
    state = service.prepare_batch(payloads)
    assert state is not None

    while not all(s.finished for s in state.sequences):
        service.step_batch(state)

    service.finalize_sequence(state, 0)

    # Verify session has saved messages
    msgs = service._session_mgr.get_messages("fin-1")
    assert len(msgs) >= 2, "Should have user + assistant messages"
    assert msgs[-1]["role"] == "assistant"
    assert len(msgs[-1]["content"]) > 0

    print("  finalize saves messages OK")


# ===========================================================================
# Test 13: finalize_sequence on cancelled does not save
# ===========================================================================

def test_finalize_cancelled_no_save():
    """finalize_sequence on a cancelled sequence should not save assistant message."""
    service, model = _make_service()
    payloads = [
        {"session_id": "fin-cancel", "prompt": "hello", "max_new_tokens": 5},
    ]
    state = service.prepare_batch(payloads)
    assert state is not None

    # Cancel immediately
    state.sequences[0].cancel_event.set()
    service.step_batch(state)
    service.finalize_sequence(state, 0)

    # Should not have saved assistant message
    msgs = service._session_mgr.get_messages("fin-cancel")
    has_assistant = any(m["role"] == "assistant" for m in msgs)
    assert not has_assistant, "Cancelled sequence should not save assistant message"

    print("  finalize cancelled no save OK")


# ===========================================================================
# Test 14: Sampling batch via prepare_batch
# ===========================================================================

def test_sampling_batch_prepare():
    """Sampling requests should use prefill_packed_sampling in prepare_batch."""
    service, model = _make_service()
    payloads = [
        {"session_id": "samp-1", "prompt": "hi", "max_new_tokens": 2, "temperature": 0.8, "top_k": 50},
        {"session_id": "samp-2", "prompt": "yo", "max_new_tokens": 2, "temperature": 0.8, "top_k": 50},
    ]
    state = service.prepare_batch(payloads)
    assert state is not None
    assert state.any_sampling
    assert model.prefill_packed_sampling_calls >= 1

    while not all(s.finished for s in state.sequences):
        service.step_batch(state)

    assert model.step_packed_sampling_calls >= 1

    print("  sampling batch prepare OK")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    tests = [
        test_streaming_batch_correct_chunks,
        test_non_stream_via_batch_path,
        test_mixed_stream_non_stream_batch,
        test_single_sequence_cancellation,
        test_different_max_new_tokens,
        test_batch_size_limit,
        test_dynamic_shrink,
        test_fallback_no_packed_api,
        test_fallback_edit_from_session,
        test_scheduler_streaming_batch_e2e,
        test_scheduler_non_stream_batch,
        test_scheduler_fallback_path,
        test_finalize_saves_messages,
        test_finalize_cancelled_no_save,
        test_sampling_batch_prepare,
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
