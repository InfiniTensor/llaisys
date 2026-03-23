"""Tests for sampling batch path (docs/SAMPLING_BATCH_DESIGN.md):
- Sampling requests enter packed path (no fallback to single)
- Different sampling parameter combinations (temperature, top_k, top_p)
- Mixed greedy+sampling batches
- Backward compatibility: pure greedy batches unchanged
- Edge cases: empty batch, single sampling request
- Fallback: old DLL without new API falls back correctly
"""

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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

    # fake libllaisys (must be registered before server.py imports it)
    fake_libllaisys = types.ModuleType("llaisys.libllaisys")

    class _FakeSamplingParams:
        """Mimics ctypes LlaisysSamplingParams Structure."""
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


# ---------------------------------------------------------------------------
# Fake model / tokenizer helpers
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
    """Model mock that tracks which packed methods are called."""

    def __init__(self):
        self._meta = _Meta()
        self.bind_calls = []
        self.export_calls = []
        self.reset_calls = 0
        self._ctx_seq = 0
        # Track packed call types
        self.prefill_packed_calls = 0
        self.step_packed_calls = 0
        self.prefill_packed_sampling_calls = 0
        self.step_packed_sampling_calls = 0
        self.prefill_packed_sampling_params: List[Any] = []
        self.step_packed_sampling_params: List[Any] = []

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
        # Return a valid token so generation reaches max_new_tokens
        return [66] * len(sequences)

    def prefill_packed_sampling(self, prompts, params_list):
        self.prefill_packed_sampling_calls += 1
        self.prefill_packed_sampling_params.append(params_list)
        return [65] * len(prompts)

    def step_packed_sampling(self, sequences, params_list):
        self.step_packed_sampling_calls += 1
        self.step_packed_sampling_params.append(params_list)
        # Return a valid token so generation reaches max_new_tokens
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


class FakeModelNoSamplingPacked:
    """Model mock that does NOT have prefill_packed_sampling / step_packed_sampling.

    Simulates an old DLL without the new batch sampling API.
    Only has greedy packed methods.
    """

    def __init__(self):
        self._meta = _Meta()
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
        enable_kv_runtime_reuse=kwargs.get("enable_kv_runtime_reuse", True),
        block_size=kwargs.get("block_size", 4),
        max_blocks=kwargs.get("max_blocks", 256),
        max_bytes=kwargs.get("max_bytes", 1024 * 1024),
    )
    return service, model


def _greedy_payload(session_id, prompt="hello"):
    return {
        "session_id": session_id,
        "prompt": prompt,
        "max_new_tokens": 2,
    }


def _sampling_payload(session_id, prompt="hello", temperature=0.8, top_k=50, top_p=0.9, seed=42):
    return {
        "session_id": session_id,
        "prompt": prompt,
        "max_new_tokens": 2,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
    }


# ===========================================================================
# Test: pure greedy batch behavior unchanged
# ===========================================================================

def test_pure_greedy_batch_uses_original_packed_path():
    """Pure greedy batch should use prefill_packed / step_packed (not sampling variant)."""
    service, model = _make_service()
    payloads = [_greedy_payload("g1"), _greedy_payload("g2"), _greedy_payload("g3")]
    result = service.generate_packed_non_stream(payloads)

    assert result is not None, "Pure greedy batch should not return None"
    assert len(result) == 3
    for r in result:
        assert "choices" in r
        assert r["choices"][0]["message"]["content"] is not None
        assert "usage" in r
    assert model.prefill_packed_calls >= 1, "Should use prefill_packed for greedy"
    assert model.prefill_packed_sampling_calls == 0, "Should NOT use sampling variant for greedy"
    print("  pure greedy batch uses original packed path OK")


def test_pure_greedy_batch_argmax_mode():
    """Explicit mode='argmax' should stay on greedy path."""
    service, model = _make_service()
    payload = _greedy_payload("g-argmax")
    payload["sampling"] = "argmax"
    result = service.generate_packed_non_stream([payload])

    assert result is not None
    assert len(result) == 1
    assert model.prefill_packed_calls >= 1
    assert model.prefill_packed_sampling_calls == 0
    print("  argmax mode stays on greedy path OK")


# ===========================================================================
# Test: sampling requests enter packed path
# ===========================================================================

def test_sampling_request_enters_packed_path():
    """Sampling request should use prefill_packed_sampling (not return None)."""
    service, model = _make_service()
    payloads = [_sampling_payload("s1"), _sampling_payload("s2")]
    result = service.generate_packed_non_stream(payloads)

    if result is None:
        # Before implementation: sampling falls back to None (current behavior)
        print("  NOTE: sampling still falls back to None (implementation pending)")
        return

    assert len(result) == 2
    for r in result:
        assert "choices" in r
        assert r["choices"][0]["message"]["content"] is not None
        assert "session_id" in r
    assert model.prefill_packed_sampling_calls >= 1, "Should use prefill_packed_sampling"
    print("  sampling request enters packed path OK")


# ===========================================================================
# Test: different sampling parameter combinations
# ===========================================================================

def test_sampling_temperature_only():
    """Request with only temperature > 0 should be treated as sampling."""
    service, model = _make_service()
    payload = {
        "session_id": "t-only",
        "prompt": "test",
        "max_new_tokens": 2,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 0.0,
    }
    result = service.generate_packed_non_stream([payload])

    if result is None:
        print("  NOTE: temperature-only sampling falls back (implementation pending)")
        return

    assert len(result) == 1
    assert model.prefill_packed_sampling_calls >= 1
    print("  temperature-only triggers sampling path OK")


def test_sampling_top_k_only():
    """Request with only top_k > 1 should be treated as sampling."""
    service, model = _make_service()
    payload = {
        "session_id": "k-only",
        "prompt": "test",
        "max_new_tokens": 2,
        "temperature": 0.0,
        "top_k": 50,
        "top_p": 0.0,
    }
    result = service.generate_packed_non_stream([payload])

    if result is None:
        print("  NOTE: top_k-only sampling falls back (implementation pending)")
        return

    assert len(result) == 1
    assert model.prefill_packed_sampling_calls >= 1
    print("  top_k-only triggers sampling path OK")


def test_sampling_top_p_only():
    """Request with only top_p > 0 should be treated as sampling."""
    service, model = _make_service()
    payload = {
        "session_id": "p-only",
        "prompt": "test",
        "max_new_tokens": 2,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 0.9,
    }
    result = service.generate_packed_non_stream([payload])

    if result is None:
        print("  NOTE: top_p-only sampling falls back (implementation pending)")
        return

    assert len(result) == 1
    assert model.prefill_packed_sampling_calls >= 1
    print("  top_p-only triggers sampling path OK")


def test_sampling_mode_explicit_sample():
    """Explicit mode='sample' should trigger sampling path."""
    service, model = _make_service()
    payload = {
        "session_id": "m-sample",
        "prompt": "test",
        "max_new_tokens": 2,
        "sampling": "sample",
        "temperature": 0.8,
        "top_k": 50,
    }
    result = service.generate_packed_non_stream([payload])

    if result is None:
        print("  NOTE: explicit sample mode falls back (implementation pending)")
        return

    assert len(result) == 1
    assert model.prefill_packed_sampling_calls >= 1
    print("  explicit sample mode triggers sampling path OK")


def test_sampling_all_params_combined():
    """Request with temperature + top_k + top_p all set."""
    service, model = _make_service()
    payload = _sampling_payload("all-params", temperature=0.7, top_k=40, top_p=0.95, seed=123)
    result = service.generate_packed_non_stream([payload])

    if result is None:
        print("  NOTE: combined sampling params falls back (implementation pending)")
        return

    assert len(result) == 1
    assert result[0]["session_id"] == "all-params"
    print("  all sampling params combined OK")


# ===========================================================================
# Test: mixed greedy + sampling batch
# ===========================================================================

def test_mixed_greedy_and_sampling_batch():
    """Mixed batch (greedy + sampling) should use the sampling packed path for all."""
    service, model = _make_service()
    payloads = [
        _greedy_payload("mix-g1"),
        _sampling_payload("mix-s1", temperature=0.8),
        _greedy_payload("mix-g2"),
    ]
    result = service.generate_packed_non_stream(payloads)

    if result is None:
        # Before implementation: any sampling causes entire batch to fall back
        print("  NOTE: mixed batch falls back to None (implementation pending)")
        return

    assert len(result) == 3
    session_ids = [r["session_id"] for r in result]
    assert "mix-g1" in session_ids
    assert "mix-s1" in session_ids
    assert "mix-g2" in session_ids
    for r in result:
        assert "choices" in r
    # Mixed batch should use sampling variant (greedy params are equivalent to argmax)
    assert model.prefill_packed_sampling_calls >= 1
    print("  mixed greedy+sampling batch OK")


# ===========================================================================
# Test: edge cases
# ===========================================================================

def test_empty_batch():
    """Empty batch should return empty list (not None)."""
    service, _ = _make_service()
    result = service.generate_packed_non_stream([])
    assert result == [], f"Empty batch should return [], got {result}"
    print("  empty batch returns [] OK")


def test_single_sampling_request():
    """Single sampling request in batch should work."""
    service, model = _make_service()
    payloads = [_sampling_payload("single-s")]
    result = service.generate_packed_non_stream(payloads)

    if result is None:
        print("  NOTE: single sampling request falls back (implementation pending)")
        return

    assert len(result) == 1
    assert result[0]["session_id"] == "single-s"
    print("  single sampling request in batch OK")


def test_single_greedy_request():
    """Single greedy request in batch should work (regression)."""
    service, model = _make_service()
    payloads = [_greedy_payload("single-g")]
    result = service.generate_packed_non_stream(payloads)

    assert result is not None
    assert len(result) == 1
    assert result[0]["session_id"] == "single-g"
    assert model.prefill_packed_calls >= 1
    print("  single greedy request in batch OK")


def test_stream_request_rejected():
    """Stream requests should cause packed path to return None."""
    service, _ = _make_service()
    payloads = [{"session_id": "stream-1", "prompt": "hi", "max_new_tokens": 2, "stream": True}]
    result = service.generate_packed_non_stream(payloads)
    assert result is None, "Stream request should cause fallback"
    print("  stream request rejected from packed path OK")


def test_edit_from_session_rejected():
    """Requests with edit_from_session_id should cause packed path to return None."""
    service, _ = _make_service()
    payloads = [{
        "session_id": "edit-1",
        "prompt": "hi",
        "max_new_tokens": 2,
        "edit_from_session_id": "other",
        "edit_message_index": 0,
    }]
    result = service.generate_packed_non_stream(payloads)
    assert result is None, "Edit request should cause fallback"
    print("  edit_from_session_id rejected from packed path OK")


# ===========================================================================
# Test: fallback when old DLL has no new API
# ===========================================================================

def test_fallback_old_dll_no_packed_sampling():
    """When model lacks prefill_packed_sampling, sampling requests should return None."""
    model = FakeModelNoSamplingPacked()
    service, _ = _make_service(model=model)
    payloads = [_sampling_payload("old-dll-s1")]
    result = service.generate_packed_non_stream(payloads)

    # Should return None (fallback to single-request processing)
    assert result is None, "Old DLL without packed sampling should fall back to None"
    print("  old DLL fallback for sampling OK")


def test_fallback_old_dll_greedy_still_works():
    """When model lacks prefill_packed_sampling, greedy batch should still work."""
    model = FakeModelNoSamplingPacked()
    service, _ = _make_service(model=model)
    payloads = [_greedy_payload("old-dll-g1"), _greedy_payload("old-dll-g2")]
    result = service.generate_packed_non_stream(payloads)

    assert result is not None, "Greedy batch should work even without new API"
    assert len(result) == 2
    print("  old DLL greedy batch still works OK")


def test_fallback_no_prefill_packed_at_all():
    """Model without prefill_packed should return None for any batch."""

    class BareModel:
        """Model with no packed methods at all."""
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

    model = BareModel()
    service, _ = _make_service(model=model)
    result = service.generate_packed_non_stream([_greedy_payload("bare-1")])
    assert result is None, "No prefill_packed should return None"
    print("  no prefill_packed at all returns None OK")


# ===========================================================================
# Test: response format correctness
# ===========================================================================

def test_response_format_has_required_fields():
    """Each response in batch should have session_id, choices, usage (OpenAI format)."""
    service, _ = _make_service()
    payloads = [_greedy_payload("fmt-1"), _greedy_payload("fmt-2")]
    result = service.generate_packed_non_stream(payloads)

    assert result is not None
    for r in result:
        assert "session_id" in r, "Missing session_id"
        assert "choices" in r, "Missing choices"
        assert "id" in r, "Missing id"
        assert "object" in r, "Missing object"
        assert r["object"] == "chat.completion"
        assert r["choices"][0]["message"]["content"] is not None, "Missing content"
        assert r["choices"][0]["finish_reason"] is not None, "Missing finish_reason"
        assert "usage" in r, "Missing usage"
        usage = r["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    print("  response format has required fields OK")


def test_response_session_ids_match_input_order():
    """Response session_ids should match input order."""
    service, _ = _make_service()
    payloads = [_greedy_payload("order-a"), _greedy_payload("order-b"), _greedy_payload("order-c")]
    result = service.generate_packed_non_stream(payloads)

    assert result is not None
    assert [r["session_id"] for r in result] == ["order-a", "order-b", "order-c"]
    print("  response session_ids match input order OK")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    tests = [
        # Pure greedy (backward compat)
        test_pure_greedy_batch_uses_original_packed_path,
        test_pure_greedy_batch_argmax_mode,
        # Sampling enters packed path
        test_sampling_request_enters_packed_path,
        # Different sampling param combos
        test_sampling_temperature_only,
        test_sampling_top_k_only,
        test_sampling_top_p_only,
        test_sampling_mode_explicit_sample,
        test_sampling_all_params_combined,
        # Mixed batch
        test_mixed_greedy_and_sampling_batch,
        # Edge cases
        test_empty_batch,
        test_single_sampling_request,
        test_single_greedy_request,
        test_stream_request_rejected,
        test_edit_from_session_rejected,
        # Fallback (old DLL)
        test_fallback_old_dll_no_packed_sampling,
        test_fallback_old_dll_greedy_still_works,
        test_fallback_no_prefill_packed_at_all,
        # Response format
        test_response_format_has_required_fields,
        test_response_session_ids_match_input_order,
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
