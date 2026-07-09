import importlib
import os
import sys
import threading
from dataclasses import dataclass


def _load_backend_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_root = os.path.join(repo_root, "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    return importlib.import_module("llaisys.backend")


@dataclass
class _FakeSlot:
    kcache_array: object = None
    vcache_array: object = None
    past_len: int = 0


class _FakePool:
    def __init__(self):
        self._committed = {}

    @staticmethod
    def _common_prefix_len(a, b):
        limit = min(len(a), len(b))
        for i in range(limit):
            if a[i] != b[i]:
                return i
        return limit

    def prepare_session(self, session_id, prompt_tokens, _max_new_tokens):
        history = self._committed.get(session_id, [])
        shared = self._common_prefix_len(history, prompt_tokens)
        return _FakeSlot(past_len=shared)

    def commit_session(self, session_id, full_tokens):
        self._committed[session_id] = list(full_tokens)

    def reset_session(self, session_id):
        self._committed.pop(session_id, None)

    def clear(self):
        self._committed.clear()


class _FakeTokenizer:
    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False):
        pieces = [f"{m['role']}:{m['content']}" for m in conversation]
        if add_generation_prompt:
            pieces.append("assistant:")
        return "\n".join(pieces)

    def encode(self, prompt):
        return [ord(ch) % 251 for ch in prompt]

    def decode(self, token_ids, skip_special_tokens=True):
        _ = skip_special_tokens
        return " ".join(str(t) for t in token_ids)


class _FakeModel:
    eos_token_id = 999999

    def __init__(self):
        self._counter = 700
        self.infer_calls = []

    def generate(self, inputs, max_new_tokens=1, **kwargs):
        _ = kwargs
        generated = list(inputs)
        for _ in range(max_new_tokens):
            self._counter += 1
            generated.append(self._counter)
        return generated

    def _infer_tokens(self, tokens, _kcache, _vcache, past_len, _temperature, _top_k, _top_p):
        self.infer_calls.append({"tokens": list(tokens), "past_len": past_len})
        self._counter += 1
        return self._counter


def _make_backend(backend_module):
    backend = backend_module.InferenceBackend.__new__(backend_module.InferenceBackend)
    backend._tokenizer = _FakeTokenizer()
    backend._model = _FakeModel()
    backend._pool = _FakePool()
    backend._sessions = {}
    backend._lock = threading.Lock()
    return backend


def test_backend_session_interfaces():
    backend_module = _load_backend_module()
    assert hasattr(backend_module, "InferenceBackend")
    assert hasattr(backend_module, "SessionState")


def test_backend_optional_scenario_and_stream():
    backend_module = _load_backend_module()
    backend = _make_backend(backend_module)

    # 会话1：who are you?
    s1 = backend.create_session("session-1")
    backend.append_message(s1.session_id, "user", "who are you?")
    result1 = backend.generate(
        session_id=s1.session_id,
        max_tokens=3,
        top_k=10,
        top_p=0.9,
        temperature=0.7,
        use_cache=True,
    )
    
    assert len(result1["completion_ids"]) == 3
    assert backend.get_session(s1.session_id).messages[-1]["role"] == "assistant"

    # 新建会话2：相同提示词 who are you?
    s2 = backend.create_session("session-2")
    backend.append_message(s2.session_id, "user", "who are you?")
    result2 = backend.generate(
        session_id=s2.session_id,
        max_tokens=2,
        top_k=10,
        top_p=0.9,
        temperature=0.7,
        use_cache=True,
    )
    assert len(result2["completion_ids"]) == 2
    
    # 删除会话1
    backend.delete_session("session-1")
    sessions = [s.session_id for s in backend.list_sessions()]
    assert "session-1" not in sessions
    assert "session-2" in sessions

    # 修改会话2历史问题并重生成
    backend.replace_messages(
        "session-2",
        [{"role": "user", "content": "who are you exactly?"}],
    )
    result3 = backend.generate(
        session_id="session-2",
        max_tokens=2,
        top_k=10,
        top_p=0.9,
        temperature=0.7,
        use_cache=True,
    )
    assert len(result3["completion_ids"]) == 2
    assert backend.get_session("session-2").messages[-1]["role"] == "assistant"
    
    # 流式生成覆盖
    backend.replace_messages(
        "session-2",
        [{"role": "user", "content": "stream test"}],
    )
    chunks = list(
        backend.stream_generate(
            session_id="session-2",
            max_tokens=4,
            top_k=10,
            top_p=0.9,
            temperature=0.7,
            use_cache=True,
        )
    )

    assert len(chunks) == 4
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(backend._model.infer_calls) == 4
    assert len(backend._model.infer_calls[0]["tokens"]) > 1
    assert all(len(call["tokens"]) == 1 for call in backend._model.infer_calls[1:])


if __name__ == "__main__":
    test_backend_session_interfaces()
    test_backend_optional_scenario_and_stream()
    print("\033[92mTest passed!\033[0m\n")
