from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, Optional, Sequence
import ctypes

from ..libllaisys import LIB_LLAISYS, llaisysTensor_t, DeviceType, DataType, MemcpyKind
from ..runtime import RuntimeAPI


@dataclass
class PrefixMatch:
    """Prefix match result from pool-level lookup.

    Attributes:
        session_id: Matched source session id.
        matched_tokens: Number of matched prefix tokens.
    """
    session_id: str
    matched_tokens: int


@dataclass
class KVCacheSlot:
    """Per-session KV cache slot.

    Attributes:
        session_id: Session identifier.
        capacity: Max tokens storable in this slot.
        kcache_array: C-level K cache tensor array, one per layer.
        vcache_array: C-level V cache tensor array, one per layer.
        tokens: Full token history tracked by upper layer.
        past_len: Valid reusable prefix length in current caches.
    """
    session_id: str
    capacity: int
    kcache_array: object
    vcache_array: object
    tokens: tuple[int, ...]
    past_len: int


class _TrieNode:
    """Trie node for token-prefix indexing.

    Attributes:
        token: Current node token value, root is None.
        children: Mapping token -> child node.
        session_indices: Sessions that contain this prefix.
    """
    __slots__ = ("token", "children", "session_indices")

    def __init__(self, token: Optional[int] = None):
        self.token = token
        self.children: dict[int, _TrieNode] = {}
        self.session_indices: list[int] = []


class KVCachePool:
    """External KV cache manager for multi-session inference.

    Notes:
    - This pool owns tensor lifecycle for created caches.
    - Prefix matching is based on token history per session.
    - Backend should call `prepare_session` before generation and
      `commit_session` after generation.
    """

    def __init__(self, model, max_sessions: int = 32):
        """Create a KV cache pool.

        Args:
            model: Qwen2 model instance that provides tensor/meta attributes.
            max_sessions: Maximum active sessions retained in memory.
        """
        self._model = model
        self._max_sessions = max_sessions
        self._slots: "OrderedDict[str, KVCacheSlot]" = OrderedDict()
        self._trie_root = _TrieNode()
        self._next_session_index = 0
        self._session_to_index: dict[str, int] = {}
        self._index_to_session: dict[int, str] = {}
        self._active_indices: set[int] = set()
        self._runtime = RuntimeAPI(self._model.device)

    @staticmethod
    def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
        """Return common-prefix length for two token sequences."""
        limit = min(len(a), len(b))
        for i in range(limit):
            if a[i] != b[i]:
                return i
        return limit

    def _destroy_slot(self, slot: KVCacheSlot) -> None:
        """Release all tensor resources owned by one slot."""
        for i in range(self._model.num_hidden_layers):
            if slot.kcache_array[i]:
                LIB_LLAISYS.tensorDestroy(slot.kcache_array[i])
            if slot.vcache_array[i]:
                LIB_LLAISYS.tensorDestroy(slot.vcache_array[i])

    @staticmethod
    def _dtype_nbytes(dtype: DataType) -> int:
        if dtype in (DataType.BYTE, DataType.BOOL, DataType.I8, DataType.U8):
            return 1
        if dtype in (DataType.I16, DataType.U16, DataType.F16, DataType.BF16):
            return 2
        if dtype in (DataType.I32, DataType.U32, DataType.F32):
            return 4
        if dtype in (DataType.I64, DataType.U64, DataType.F64):
            return 8
        raise ValueError(f"unsupported dtype for KV cache copy: {dtype}")

    def _copy_prefix_cache(self, src_slot: KVCacheSlot, dst_slot: KVCacheSlot, prefix_len: int) -> None:
        if prefix_len <= 0:
            return

        nbytes = (
            prefix_len
            * self._model.num_key_value_heads
            * self._model.per_kvhead_dim
            * self._dtype_nbytes(self._model.data_type)
        )
        memcpy_kind = MemcpyKind.H2H if self._model.device == DeviceType.CPU else MemcpyKind.D2D

        for i in range(self._model.num_hidden_layers):
            src_k = LIB_LLAISYS.tensorSlice(src_slot.kcache_array[i], 0, 0, prefix_len)
            dst_k = LIB_LLAISYS.tensorSlice(dst_slot.kcache_array[i], 0, 0, prefix_len)
            src_v = LIB_LLAISYS.tensorSlice(src_slot.vcache_array[i], 0, 0, prefix_len)
            dst_v = LIB_LLAISYS.tensorSlice(dst_slot.vcache_array[i], 0, 0, prefix_len)
            try:
                self._runtime.memcpy_sync(
                    LIB_LLAISYS.tensorGetData(dst_k),
                    LIB_LLAISYS.tensorGetData(src_k),
                    nbytes,
                    memcpy_kind,
                )
                self._runtime.memcpy_sync(
                    LIB_LLAISYS.tensorGetData(dst_v),
                    LIB_LLAISYS.tensorGetData(src_v),
                    nbytes,
                    memcpy_kind,
                )
            finally:
                LIB_LLAISYS.tensorDestroy(src_k)
                LIB_LLAISYS.tensorDestroy(dst_k)
                LIB_LLAISYS.tensorDestroy(src_v)
                LIB_LLAISYS.tensorDestroy(dst_v)

    def _get_or_create_session_index(self, session_id: str) -> int:
        """Get stable integer index for a session id."""
        idx = self._session_to_index.get(session_id)
        if idx is not None:
            return idx
        idx = self._next_session_index
        self._next_session_index += 1
        self._session_to_index[session_id] = idx
        self._index_to_session[idx] = session_id
        return idx

    @staticmethod
    def _add_session_index(node: _TrieNode, session_index: int) -> None:
        """Attach one session index to trie node if absent."""
        if session_index not in node.session_indices:
            node.session_indices.append(session_index)

    @staticmethod
    def _remove_session_index(node: _TrieNode, session_index: int) -> None:
        """Detach one session index from trie node if present."""
        if session_index in node.session_indices:
            node.session_indices.remove(session_index)

    def _trie_insert(self, tokens: Sequence[int], session_index: int) -> None:
        """Insert a token prefix path into trie for one session."""
        node = self._trie_root
        for token in tokens:
            child = node.children.get(token)
            if child is None:
                child = _TrieNode(token)
                node.children[token] = child
            node = child
            self._add_session_index(node, session_index)

    def _trie_remove(self, tokens: Sequence[int], session_index: int) -> None:
        """Remove one session's token prefix path from trie."""
        if not tokens:
            return

        stack: list[tuple[_TrieNode, int, _TrieNode]] = []
        node = self._trie_root
        for token in tokens:
            child = node.children.get(token)
            if child is None:
                return
            stack.append((node, token, child))
            node = child

        for _, _, child in stack:
            self._remove_session_index(child, session_index)

        for parent, token, child in reversed(stack):
            if child.children or child.session_indices:
                break
            del parent.children[token]

    def _trie_remove_from_depth(self, tokens: Sequence[int], session_index: int, keep_depth: int) -> None:
        """Remove session index from trie nodes after `keep_depth` on one token path.

        Args:
            tokens: Existing cached token path.
            session_index: Target session index.
            keep_depth: Number of leading nodes to keep unchanged.
        """
        if not tokens:
            return

        stack: list[tuple[_TrieNode, int, _TrieNode]] = []
        node = self._trie_root
        for token in tokens:
            child = node.children.get(token)
            if child is None:
                return
            stack.append((node, token, child))
            node = child

        for depth, (_, _, child) in enumerate(stack, start=1):
            if depth > keep_depth:
                self._remove_session_index(child, session_index)

        for parent, token, child in reversed(stack):
            if child.children or child.session_indices:
                break
            del parent.children[token]

    def _trie_best_prefix_len(self, tokens: Sequence[int], exclude_session_id: Optional[str]) -> tuple[int, Optional[str]]:
        """Find longest matched prefix and source session by trie traversal."""
        exclude_index = None
        if exclude_session_id is not None:
            exclude_index = self._session_to_index.get(exclude_session_id)

        best_len = 0
        best_session_id: Optional[str] = None
        node = self._trie_root

        for depth, token in enumerate(tokens, start=1):
            node = node.children.get(token)
            if node is None:
                break

            matched_session_id = None
            for session_index in node.session_indices:
                if session_index == exclude_index:
                    continue
                if session_index not in self._active_indices:
                    continue
                sid = self._index_to_session.get(session_index)
                if sid is None or sid not in self._slots:
                    continue
                matched_session_id = sid
                break

            if matched_session_id is not None:
                best_len = depth
                best_session_id = matched_session_id

        return best_len, best_session_id

    def _update_slot_tokens(self, session_id: str, new_tokens: Sequence[int]) -> None:
        """Update a session slot token history and synchronize trie index."""
        slot = self._slots.get(session_id)
        if slot is None:
            return

        session_index = self._get_or_create_session_index(session_id)
        self._active_indices.add(session_index)

        old_tokens = slot.tokens[:slot.past_len]
        if old_tokens:
            self._trie_remove(old_tokens, session_index)

        slot.tokens = tuple(new_tokens)
        slot.past_len = len(slot.tokens)

        if slot.past_len > 0:
            self._trie_insert(slot.tokens[:slot.past_len], session_index)

    def _set_slot_reuse_state(self, session_id: str, new_tokens: Sequence[int], reusable_len: int) -> KVCacheSlot:
        slot = self._slots.get(session_id)
        if slot is None:
            raise KeyError(f"session not found: {session_id}")

        new_tokens = tuple(new_tokens)
        reusable_len = max(0, min(reusable_len, len(new_tokens)))

        session_index = self._get_or_create_session_index(session_id)
        self._active_indices.add(session_index)

        old_cached_tokens = slot.tokens[:slot.past_len]
        if old_cached_tokens:
            self._trie_remove(old_cached_tokens, session_index)

        slot.tokens = new_tokens
        slot.past_len = reusable_len

        if reusable_len > 0:
            self._trie_insert(new_tokens[:reusable_len], session_index)

        return slot

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used sessions when exceeding pool capacity."""
        while len(self._slots) > self._max_sessions:
            session_id, slot = self._slots.popitem(last=False)
            session_index = self._session_to_index.get(session_id)
            if session_index is not None:
                if slot.past_len > 0:
                    self._trie_remove(slot.tokens[:slot.past_len], session_index)
                self._active_indices.discard(session_index)
            self._destroy_slot(slot)

    def _allocate_slot(self, session_id: str, capacity: int) -> KVCacheSlot:
        """Allocate one session slot and its per-layer KV tensors."""
        array_type = llaisysTensor_t * self._model.num_hidden_layers
        kcache_array = array_type()
        vcache_array = array_type()

        for i in range(self._model.num_hidden_layers):
            shape_arr = (ctypes.c_size_t * 3)(
                capacity,
                self._model.num_key_value_heads,
                self._model.per_kvhead_dim,
            )
            kcache_array[i] = LIB_LLAISYS.tensorCreate(
                shape_arr, 3, self._model.data_type, self._model.device, self._model.device_id
            )
            vcache_array[i] = LIB_LLAISYS.tensorCreate(
                shape_arr, 3, self._model.data_type, self._model.device, self._model.device_id
            )

        return KVCacheSlot(
            session_id=session_id,
            capacity=capacity,
            kcache_array=kcache_array,
            vcache_array=vcache_array,
            tokens=(),
            past_len=0,
        )

    def find_best_prefix(self, tokens: Sequence[int], exclude_session_id: Optional[str] = None) -> Optional[PrefixMatch]:
        """Find best reusable prefix from pool.

        Args:
            tokens: Target token sequence.
            exclude_session_id: Optional session id to skip.

        Returns:
            PrefixMatch for longest hit, or None if no positive-length match.
        """
        matched_len, matched_session = self._trie_best_prefix_len(tokens, exclude_session_id)
        if matched_session is None or matched_len == 0:
            return None
        return PrefixMatch(session_id=matched_session, matched_tokens=matched_len)

    def prepare_session(self, session_id: str, prompt_tokens: Sequence[int], max_new_tokens: int) -> KVCacheSlot:
        """Prepare slot for a new inference call.

        This method ensures slot existence/capacity, computes best reusable
        prefix from current session and pool-level matches, and returns slot
        handles for model inference.

        Args:
            session_id: Target session id.
            prompt_tokens: Full prompt tokens for this call.
            max_new_tokens: Planned generation upper bound.

        Returns:
            Prepared KVCacheSlot with updated `past_len`.
        """
        prompt_tokens = tuple(prompt_tokens)
        if not prompt_tokens:
            raise ValueError("prompt_tokens cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        required_capacity = len(prompt_tokens) + max_new_tokens
        slot = self._slots.get(session_id)
        existed_before = slot is not None

        if slot is None:
            slot = self._allocate_slot(session_id, max(required_capacity, self._model.max_position_embeddings))
            self._slots[session_id] = slot
        elif slot.capacity < required_capacity:
            self._destroy_slot(slot)
            slot = self._allocate_slot(session_id, max(required_capacity, self._model.max_position_embeddings))
            self._slots[session_id] = slot

        self_shared = self._common_prefix_len(slot.tokens[:slot.past_len], prompt_tokens) if existed_before else 0

        pool_match = self.find_best_prefix(prompt_tokens, exclude_session_id=session_id)
        pool_shared = pool_match.matched_tokens if pool_match is not None else 0

        chosen_shared = self_shared
        source_slot: Optional[KVCacheSlot] = None
        if pool_shared > chosen_shared and pool_match is not None:
            source_slot = self._slots.get(pool_match.session_id)
            if source_slot is not None:
                chosen_shared = pool_shared

        slot = self._set_slot_reuse_state(session_id, prompt_tokens, chosen_shared)
        if source_slot is not None and chosen_shared > 0:
            self._copy_prefix_cache(source_slot, slot, chosen_shared)

        self._slots.move_to_end(session_id)
        self._evict_if_needed()
        return slot

    def modify_session_tokens(self, session_id: str, edited_tokens: Sequence[int]) -> KVCacheSlot:
        """Apply history edit for one session and invalidate cache suffix.

        The pool keeps trie/session state for common prefix unchanged and drops
        session index on trie nodes after the first edited position.

        Args:
            session_id: Target session id.
            edited_tokens: New full token history after edit.

        Returns:
            Updated KVCacheSlot whose `past_len` is the reusable cache prefix.

        Raises:
            KeyError: If target session does not exist.
        """
        slot = self._slots.get(session_id)
        if slot is None:
            raise KeyError(f"session not found: {session_id}")

        new_tokens = tuple(edited_tokens)
        old_cached_tokens = slot.tokens[:slot.past_len]
        keep_len = self._common_prefix_len(old_cached_tokens, new_tokens)

        session_index = self._get_or_create_session_index(session_id)
        self._active_indices.add(session_index)

        if old_cached_tokens:
            self._trie_remove_from_depth(old_cached_tokens, session_index, keep_len)
        if keep_len > 0:
            self._trie_insert(new_tokens[:keep_len], session_index)

        slot.tokens = new_tokens
        slot.past_len = keep_len
        self._slots.move_to_end(session_id)
        return slot

    def commit_session(self, session_id: str, full_tokens: Sequence[int]) -> None:
        """Commit generation result and refresh trie index.

        Args:
            session_id: Target session id.
            full_tokens: Full tokens after generation (prompt + output).
        """
        slot = self._slots.get(session_id)
        if slot is None:
            return
        self._update_slot_tokens(session_id, tuple(full_tokens))
        self._slots.move_to_end(session_id)

    def reset_session(self, session_id: str) -> None:
        """Remove one session and release its cache resources."""
        slot = self._slots.pop(session_id, None)
        if slot is None:
            return
        session_index = self._session_to_index.get(session_id)
        if session_index is not None:
            if slot.past_len > 0:
                self._trie_remove(slot.tokens[:slot.past_len], session_index)
            self._active_indices.discard(session_index)
        self._destroy_slot(slot)

    def clear(self) -> None:
        """Clear all sessions and release all cache resources."""
        for slot in self._slots.values():
            self._destroy_slot(slot)
        self._slots.clear()
        self._trie_root = _TrieNode()
        self._active_indices.clear()
