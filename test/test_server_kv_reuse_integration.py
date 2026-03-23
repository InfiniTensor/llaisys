import importlib.util
import sys
import types
from pathlib import Path


def _load_server_module():
    root = Path(__file__).resolve().parents[1]
    interfaces_path = root / "python" / "llaisys" / "interfaces.py"
    kv_path = root / "python" / "llaisys" / "kv_cache_pool.py"
    scheduler_path = root / "python" / "llaisys" / "scheduler.py"
    session_mgr_path = root / "python" / "llaisys" / "session_manager.py"
    kv_bridge_path = root / "python" / "llaisys" / "kv_runtime_bridge.py"
    server_path = root / "python" / "llaisys" / "server.py"

    # Load interfaces first (kv_cache_pool and server import from it)
    iface_spec = importlib.util.spec_from_file_location("llaisys.interfaces", str(interfaces_path))
    if iface_spec is not None and iface_spec.loader is not None:
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
    fake_llaisys.Tokenizer = object
    if session_mgr_mod:
        fake_llaisys.session_manager = session_mgr_mod
    if kv_bridge_mod:
        fake_llaisys.kv_runtime_bridge = kv_bridge_mod
    sys.modules["llaisys"] = fake_llaisys
    sys.modules["llaisys.kv_cache_pool"] = kv_mod
    sys.modules["llaisys.scheduler"] = scheduler_mod

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
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text]

    def decode(self, token_ids):
        return "".join(chr(int(t)) for t in token_ids)


class _EndToken:
    def __init__(self, value):
        self.value = value


class _Meta:
    def __init__(self):
        self.end_token = _EndToken(-1)


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
        return 65  # "A"

    def prefill_sampling(self, prompt_ids, top_k=1, top_p=0.0, temperature=0.0, seed=0):
        return self.prefill(prompt_ids)

    def step(self, token_ids):
        return 66  # "B"

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


def _make_service():
    server_mod = _load_server_module()
    model = FakeModel()
    tok = FakeTokenizer()
    service = server_mod.ChatService(
        model=model,
        tokenizer=tok,
        model_path=None,
        enable_kv_runtime_reuse=True,
        block_size=4,
        max_blocks=256,
        max_bytes=1024 * 1024,
    )
    return service, model


def test_kv_reuse_same_session_binds_native_context():
    service, model = _make_service()

    first = service.generate({"session_id": "s1", "prompt": "你好", "max_new_tokens": 2})
    assert first["session_id"] == "s1"
    # first request has no prefix hit; should bind None
    assert model.bind_calls and model.bind_calls[0] is None
    assert len(model.export_calls) == 1

    service.generate({"session_id": "s1", "prompt": "继续", "max_new_tokens": 2})
    # second request should bind non-null native context
    assert model.bind_calls[-1] is not None
    dbg = service.kv_debug_snapshot("s1")
    assert dbg["last_bind"]["bound"] is True
    assert dbg["last_bind"]["source_session_id"] == "s1"
    assert dbg["last_bind"]["prefix_len"] > 0


def test_kv_reuse_cross_session_can_use_donor_context():
    service, _ = _make_service()

    service.generate({"session_id": "donor", "prompt": "同一个问题", "max_new_tokens": 2})
    service.generate(
        {
            "session_id": "receiver",
            "messages": [{"role": "user", "content": "同一个问题"}],
            "max_new_tokens": 2,
        }
    )

    dbg = service.kv_debug_snapshot("receiver")
    assert dbg["last_bind"]["bound"] is True
    assert dbg["last_bind"]["prefix_len"] > 0
    assert dbg["last_bind"]["source_session_id"] == "donor"


def test_cancelled_request_does_not_export_native_kv():
    service, model = _make_service()

    def _cancelled_iter(prompt_ids, max_new_tokens, sampling, prefix_len, cancel_event):
        cancel_event.set()
        if False:
            yield 0

    service._iter_generate_ids = _cancelled_iter
    result = service.generate({"session_id": "s-cancel", "prompt": "会取消", "max_new_tokens": 2})
    assert result.get("stopped") is True
    assert result["choices"][0]["finish_reason"] == "stop"
    assert len(model.export_calls) == 0


if __name__ == "__main__":
    test_kv_reuse_same_session_binds_native_context()
    test_kv_reuse_cross_session_can_use_donor_context()
    test_cancelled_request_does_not_export_native_kv()
    print("server kv reuse integration tests passed")

