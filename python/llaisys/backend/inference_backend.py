import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from transformers import AutoTokenizer

import llaisys
from llaisys.models.kvcachepool import KVCachePool


@dataclass
class SessionState:
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))


class InferenceBackend:
    """Project #3 optional backend for session-aware inference.

    Responsibilities:
    - Manage multi-session message history
    - Integrate Qwen2 model with external KVCachePool
    - Support history edit + regenerate workflow
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: llaisys.DeviceType = llaisys.DeviceType.CPU,
        max_sessions: int = 32,
    ) -> None:
        tokenizer_source = model_path or llaisys.models.Qwen2.DEFAULT_MODEL_ID
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        self._model = llaisys.models.Qwen2(model_path=model_path, device=device)
        self._pool = KVCachePool(self._model, max_sessions=max_sessions)

        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None) -> SessionState:
        with self._lock:
            sid = session_id or f"sess-{uuid.uuid4().hex}"
            if sid in self._sessions:
                raise ValueError(f"session already exists: {sid}")
            state = SessionState(session_id=sid, messages=list(messages or []))
            self._sessions[sid] = state
            return state

    def list_sessions(self) -> List[SessionState]:
        with self._lock:
            return list(self._sessions.values())

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"session not found: {session_id}")
            return state

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is None:
                return
            self._pool.reset_session(session_id)

    def clear_sessions(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._pool.clear()

    def append_message(self, session_id: str, role: str, content: str) -> SessionState:
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported role: {role}")
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"session not found: {session_id}")
            state.messages.append({"role": role, "content": content})
            state.updated_at = int(time.time())
            return state

    def replace_messages(self, session_id: str, messages: List[Dict[str, str]]) -> SessionState:
        """Replace full history for one session.

        This is suitable for edit-history + regenerate workflow.
        KV cache validity will be recalculated on next generate call.
        """
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"session not found: {session_id}")
            state.messages = list(messages)
            state.updated_at = int(time.time())
            return state

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    @staticmethod
    def _normalize_think_for_session(prompt: str, completion_text: str) -> str:
        if not completion_text:
            return completion_text
        prompt_has_think = ("<think>" in prompt) or ("</think>" in prompt)
        if not prompt_has_think:
            return completion_text

        has_open = "<think>" in completion_text
        has_close = "</think>" in completion_text
        if has_close and not has_open:
            completion_text = f"<think>{completion_text}"
            has_open = True
        if has_open and not has_close:
            completion_text = f"{completion_text}</think>"
        return completion_text

    def generate(
        self,
        session_id: str,
        max_tokens: int,
        top_k: int,
        top_p: float,
        temperature: float,
        use_cache: bool = True,
        append_assistant_message: bool = True,
    ) -> Dict[str, Any]:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"session not found: {session_id}")

            prompt = self._build_prompt(state.messages)
            prompt_ids = self._tokenizer.encode(prompt)

            if not use_cache:
                output_ids = self._model.generate(
                    prompt_ids,
                    max_new_tokens=max_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    use_cache=False,
                )
                completion_ids = output_ids[len(prompt_ids):]
                self._pool.reset_session(session_id)
            else:
                slot = self._pool.prepare_session(session_id, prompt_ids, max_tokens)

                if slot.past_len == 0:
                    model_input_ids = prompt_ids
                    model_past_len = 0
                else:
                    model_input_ids = prompt_ids[slot.past_len:]
                    model_past_len = slot.past_len
                    if not model_input_ids:
                        model_input_ids = [prompt_ids[-1]]
                        model_past_len = max(0, slot.past_len - 1)

                output_ids = self._model.generate(
                    model_input_ids,
                    max_new_tokens=max_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    use_cache=True,
                    kcache_array=slot.kcache_array,
                    vcache_array=slot.vcache_array,
                    past_len=model_past_len,
                )

                completion_ids = output_ids[len(model_input_ids):]
                full_output_ids = list(prompt_ids) + list(completion_ids)
                self._pool.commit_session(session_id, full_output_ids)

            if self._model.eos_token_id in completion_ids:
                completion_ids = completion_ids[:completion_ids.index(self._model.eos_token_id)]

            completion_text = self._tokenizer.decode(completion_ids, skip_special_tokens=False)
            if append_assistant_message:
                stored_text = self._normalize_think_for_session(prompt, completion_text)
                state.messages.append({"role": "assistant", "content": stored_text})
                state.updated_at = int(time.time())

            return {
                "session_id": session_id,
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "completion_text": completion_text,
                "created": int(time.time()),
            }

    def stream_generate(
        self,
        session_id: str,
        max_tokens: int,
        top_k: int,
        top_p: float,
        temperature: float,
        use_cache: bool = True,
        append_assistant_message: bool = True,
    ) -> Iterator[str]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k == 0:
            raise ValueError("top_k cannot be 0")
        if top_p < 0 or top_p > 1:
            raise ValueError("top_p must be in [0, 1]")

        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"session not found: {session_id}")

            prompt = self._build_prompt(state.messages)
            prompt_ids = self._tokenizer.encode(prompt)
            completion_ids: List[int] = []

            if not use_cache:
                generated_ids = list(prompt_ids)
                for _ in range(max_tokens):
                    token_id = self._model._infer_tokens(
                        generated_ids,
                        None,
                        None,
                        0,
                        temperature,
                        top_k,
                        top_p,
                    )
                    if token_id == self._model.eos_token_id:
                        break
                    completion_ids.append(token_id)
                    generated_ids.append(token_id)
                    piece = self._tokenizer.decode([token_id], skip_special_tokens=False)
                    if piece:
                        yield piece
                self._pool.reset_session(session_id)
            else:
                slot = self._pool.prepare_session(session_id, prompt_ids, max_tokens)

                if slot.past_len == 0:
                    prefill_input_ids = prompt_ids
                    prefill_past_len = 0
                else:
                    prefill_input_ids = prompt_ids[slot.past_len:]
                    prefill_past_len = slot.past_len
                    if not prefill_input_ids:
                        prefill_input_ids = [prompt_ids[-1]]
                        prefill_past_len = max(0, slot.past_len - 1)

                token_id = self._model._infer_tokens(
                    prefill_input_ids,
                    slot.kcache_array,
                    slot.vcache_array,
                    prefill_past_len,
                    temperature,
                    top_k,
                    top_p,
                )
                if token_id != self._model.eos_token_id:
                    completion_ids.append(token_id)
                    piece = self._tokenizer.decode([token_id], skip_special_tokens=False)
                    if piece:
                        yield piece
                else:
                    token_id = None

                cached_token_len = prefill_past_len + len(prefill_input_ids)
                for _ in range(max_tokens - 1):
                    if token_id is None:
                        break
                    token_id = self._model._infer_tokens(
                        [token_id],
                        slot.kcache_array,
                        slot.vcache_array,
                        cached_token_len,
                        temperature,
                        top_k,
                        top_p,
                    )
                    if token_id == self._model.eos_token_id:
                        break
                    completion_ids.append(token_id)
                    cached_token_len += 1
                    piece = self._tokenizer.decode([token_id], skip_special_tokens=False)
                    if piece:
                        yield piece

                full_output_ids = list(prompt_ids) + completion_ids
                self._pool.commit_session(session_id, full_output_ids)

            if append_assistant_message:
                completion_text = self._tokenizer.decode(completion_ids, skip_special_tokens=False)
                stored_text = self._normalize_think_for_session(prompt, completion_text)
                state.messages.append({"role": "assistant", "content": stored_text})
                state.updated_at = int(time.time())
