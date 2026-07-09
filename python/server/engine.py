import threading
import time
from typing import List, Dict, Any, Iterator, Optional

import llaisys

from .config import CONFIG


class ChatEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backend = llaisys.backend.InferenceBackend(
            model_path=CONFIG.model_path,
            device=llaisys.DeviceType.CPU if CONFIG.device == "cpu" else llaisys.DeviceType.NVIDIA,
        )
        self._default_session_id = "engine-default"
        self._backend.create_session(self._default_session_id)
        self._tokenizer = self._backend._tokenizer

    def _reset_default_session(self, messages: List[Dict[str, str]]) -> None:
        self._backend.delete_session(self._default_session_id)
        self._backend.create_session(self._default_session_id, messages=list(messages))

    def _ensure_named_session(self, session_id: str, messages: List[Dict[str, str]]) -> None:
        try:
            self._backend.get_session(session_id)
            self._backend.replace_messages(session_id, messages)
        except KeyError:
            self._backend.create_session(session_id, messages=list(messages))

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)

    def generate(self, messages: List[Dict[str, str]], max_tokens: int, top_k: int, top_p: float, temperature: float, use_cache: bool,
                 session_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            active_session_id = session_id or self._default_session_id
            if session_id:
                self._ensure_named_session(active_session_id, messages)
            else:
                self._reset_default_session(messages)
            result = self._backend.generate(
                session_id=active_session_id,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                append_assistant_message=True,
            )

        return {
            "prompt_ids": result["prompt_ids"],
            "completion_ids": result["completion_ids"],
            "completion_text": result["completion_text"],
            "created": int(time.time()),
        }

    def stream_generate(self, messages: List[Dict[str, str]], max_tokens: int, top_k: int, top_p: float, temperature: float, use_cache: bool,
                        session_id: Optional[str] = None) -> Iterator[str]:
        with self._lock:
            active_session_id = session_id or self._default_session_id
            if session_id:
                self._ensure_named_session(active_session_id, messages)
            else:
                self._reset_default_session(messages)
            for token_id in self._backend.stream_generate(
                session_id=active_session_id,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                append_assistant_message=True,
            ):
                yield token_id

    def get_session_messages(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            state = self._backend.get_session(session_id)
            return list(state.messages)

    def create_session(self, session_id: Optional[str] = None) -> str:
        with self._lock:
            state = self._backend.create_session(session_id=session_id)
            return state.session_id

    def list_sessions(self) -> List[str]:
        with self._lock:
            return [item.session_id for item in self._backend.list_sessions()]

    def delete_session(self, session_id: str) -> None:
        if session_id == self._default_session_id:
            return
        with self._lock:
            self._backend.delete_session(session_id)


ENGINE = ChatEngine()
