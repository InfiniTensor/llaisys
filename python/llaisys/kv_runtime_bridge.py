from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple


class KVRuntimeBridge:
    """Bridge for native C++ KV context lifecycle (bind, export, find, release, debug)."""

    def __init__(self, model: Any, enabled: bool = False) -> None:
        self._model = model
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._native_kv_contexts: Dict[str, Any] = {}
        self._native_kv_tokens: Dict[str, Tuple[int, ...]] = {}
        self._last_kv_bind_debug: Dict[str, Dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def bind_for_request(
        self,
        context_id: str,
        prompt_ids: List[int],
        prefix_len: int,
    ) -> None:
        """Bind the best KV context to the model for the current request.

        Search order:
        1. Same context_id native context
        2. Prefix-matching donor context
        3. No match -> set_kv_context(None)
        """
        debug: Dict[str, Any] = {
            "enabled": self._enabled,
            "session_id": context_id,
            "prefix_len": int(prefix_len),
            "bound": False,
            "source_session_id": None,
            "set_kv_context_rc": None,
        }
        if not self._enabled or prefix_len <= 0:
            self._model.set_kv_context(None)
            with self._lock:
                self._last_kv_bind_debug[context_id] = debug
            return
        with self._lock:
            ctx = self._native_kv_contexts.get(context_id)
        source_session_id: Optional[str] = context_id if ctx else None
        if not ctx:
            source_session_id, ctx = self._find_for_prefix(prompt_ids, prefix_len)
            if not ctx:
                self._model.set_kv_context(None)
                with self._lock:
                    self._last_kv_bind_debug[context_id] = debug
                return
        rc = self._model.set_kv_context(ctx)
        debug["set_kv_context_rc"] = int(rc)
        debug["source_session_id"] = source_session_id
        if rc != 0:
            self._model.set_kv_context(None)
        else:
            debug["bound"] = True
        with self._lock:
            self._last_kv_bind_debug[context_id] = debug

    def export_after_request(
        self,
        context_id: str,
        tokens: List[int],
        block_size: int,
    ) -> None:
        """Export KV context after request completion for future reuse."""
        if not self._enabled:
            return
        with self._lock:
            ctx = self._native_kv_contexts.get(context_id)
        if not ctx:
            ctx = self._model.kv_context_create()
            if not ctx:
                return
            with self._lock:
                self._native_kv_contexts[context_id] = ctx
        rc = self._model.export_kv_context(ctx, block_size)
        if rc == 0:
            with self._lock:
                self._native_kv_tokens[context_id] = tuple(int(t) for t in tokens)

    def release(self, context_id: str) -> None:
        """Release native KV context for a given session."""
        with self._lock:
            ctx = self._native_kv_contexts.pop(context_id, None)
            self._native_kv_tokens.pop(context_id, None)
            self._last_kv_bind_debug.pop(context_id, None)
        if ctx:
            self._model.kv_context_release(ctx)

    def debug_snapshot(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Return KV runtime debug information."""
        with self._lock:
            if session_id:
                last_bind = dict(self._last_kv_bind_debug.get(session_id, {}))
                native_tokens = len(self._native_kv_tokens.get(session_id, ()))
                has_native_ctx = session_id in self._native_kv_contexts
            else:
                last_bind = {}
                native_tokens = 0
                has_native_ctx = False
            native_contexts = len(self._native_kv_contexts)
            tracked_token_sessions = len(self._native_kv_tokens)
        return {
            "session_id": session_id,
            "has_native_context": has_native_ctx,
            "native_tokens": native_tokens,
            "native_contexts": native_contexts,
            "tracked_token_sessions": tracked_token_sessions,
            "last_bind": last_bind,
        }

    def _find_for_prefix(
        self, prompt_ids: List[int], prefix_len: int
    ) -> Tuple[Optional[str], Any]:
        """Find native KV context matching the given prefix."""
        if prefix_len <= 0:
            return None, None
        prompt_prefix = tuple(prompt_ids[:prefix_len])
        with self._lock:
            best_sid: Optional[str] = None
            best_ctx: Any = None
            best_len = -1
            for sid, ctx in self._native_kv_contexts.items():
                tokens = self._native_kv_tokens.get(sid, ())
                tlen = len(tokens)
                if tlen < prefix_len:
                    continue
                if tuple(tokens[:prefix_len]) != prompt_prefix:
                    continue
                if tlen > best_len:
                    best_len = tlen
                    best_sid = sid
                    best_ctx = ctx
            return best_sid, best_ctx
