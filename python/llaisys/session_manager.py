from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Tuple


class SessionManager:
    """Session message history and cancellation event management."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._context_messages: Dict[str, List[Dict[str, str]]] = {}
        self._cancel_events: Dict[str, threading.Event] = {}

    def extract_messages(self, payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        """Extract context_id and message list from payload.

        Handles three input modes:
        - edit_from_session_id: branch from history edit
        - messages: direct message list
        - prompt: append to existing session history
        """
        context_id = str(payload.get("session_id") or "").strip() or str(uuid.uuid4())
        messages = payload.get("messages")
        prompt = payload.get("prompt")
        edit_from = str(payload.get("edit_from_session_id") or "").strip()
        edit_index_raw = payload.get("edit_message_index")

        if edit_from:
            with self._lock:
                source = list(self._context_messages.get(edit_from, []))
            if not source:
                raise ValueError("edit_from_session_id not found")
            if prompt is None:
                raise ValueError("prompt is required when editing history")
            if edit_index_raw is None:
                raise ValueError("edit_message_index is required when editing history")
            edit_index = int(edit_index_raw)
            if edit_index < 0 or edit_index >= len(source):
                raise ValueError("edit_message_index out of range")
            if source[edit_index].get("role") != "user":
                raise ValueError("edit_message_index must point to a user message")
            branched = source[: edit_index + 1]
            branched[edit_index] = {"role": "user", "content": str(prompt)}
            if not str(payload.get("session_id") or "").strip():
                context_id = str(uuid.uuid4())
            return context_id, branched

        if messages is not None:
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            return context_id, list(messages)

        if prompt is None:
            raise ValueError("payload must include messages or prompt")

        with self._lock:
            history = list(self._context_messages.get(context_id, []))
        history.append({"role": "user", "content": str(prompt)})
        return context_id, history

    def save_messages(self, context_id: str, messages: List[Dict[str, str]]) -> None:
        """Save session message history."""
        with self._lock:
            self._context_messages[context_id] = list(messages)

    def get_messages(self, context_id: str) -> List[Dict[str, str]]:
        """Get session message history (returns a copy)."""
        with self._lock:
            return list(self._context_messages.get(context_id, []))

    def get_cancel_event(self, context_id: str) -> threading.Event:
        """Get or create a cancellation event for the given context."""
        with self._lock:
            event = self._cancel_events.get(context_id)
            if event is None:
                event = threading.Event()
                self._cancel_events[context_id] = event
            return event

    def request_stop(self, context_id: str) -> bool:
        """Set the cancellation event for the given context."""
        with self._lock:
            event = self._cancel_events.get(context_id)
            if event is None:
                event = threading.Event()
                self._cancel_events[context_id] = event
            event.set()
            return True

    def clear_stop(self, context_id: str) -> None:
        """Clear the cancellation event for the given context."""
        with self._lock:
            event = self._cancel_events.get(context_id)
            if event:
                event.clear()
