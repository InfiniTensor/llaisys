from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

from llaisys.interfaces import IKVCachePool


@dataclass
class KVBlock:
    block_id: int
    generation: int
    parent_id: Optional[int]
    tokens: Tuple[int, ...]
    sealed: bool
    ref_count: int
    last_access: float
    prefix_key: Optional[Tuple[int, ...]]

    @property
    def size_bytes(self) -> int:
        # int64 token ids
        return len(self.tokens) * 8


@dataclass
class ContextState:
    block_ids: List[int]
    tokens: Tuple[int, ...]
    updated_at: float


@dataclass
class AcquireResult:
    context_id: str
    prefix_len: int


class KVCachePool(IKVCachePool):
    """In-memory token-block cache pool with reference counting.

    Notes:
    - Only sealed (full) blocks are indexed for cross-context sharing.
    - Block IDs are monotonic and never reused.
    """

    def __init__(
        self,
        block_size: int = 64,
        max_blocks: int = 4096,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        self._block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.max_bytes = int(max_bytes)

        self._lock = threading.Lock()
        self._next_block_id = 1
        self._blocks: Dict[int, KVBlock] = {}
        self._contexts: Dict[str, ContextState] = {}
        # prefix(tuple(tokens up to this block)) -> (block_id, generation)
        self._prefix_index: Dict[Tuple[int, ...], Tuple[int, int]] = {}
        self._total_bytes = 0
        self._acquire_count = 0
        self._prefix_hit_count = 0
        self._matched_tokens_total = 0

    @property
    def block_size(self) -> int:
        return self._block_size

    def acquire_context(self, context_id: str, tokens: Sequence[int]) -> AcquireResult:
        """Bind context to current prompt tokens.

        Returns matched prefix length for runtime reuse decision.
        """
        token_tuple = tuple(int(t) for t in tokens)
        with self._lock:
            _, matched_len = self._build_or_replace_context(context_id, token_tuple)
            self._acquire_count += 1
            self._matched_tokens_total += matched_len
            if matched_len > 0:
                self._prefix_hit_count += 1
            return AcquireResult(context_id=context_id, prefix_len=matched_len)

    def update_context(self, context_id: str, tokens: Sequence[int]) -> None:
        """Update context after generation to preserve longer prefixes."""
        token_tuple = tuple(int(t) for t in tokens)
        with self._lock:
            self._build_or_replace_context(context_id, token_tuple)

    def release_context(self, context_id: str) -> None:
        with self._lock:
            old_state = self._contexts.pop(context_id, None)
            if not old_state:
                return
            self._decref_chain(old_state.block_ids)
            self._evict_if_needed()

    def _build_or_replace_context(self, context_id: str, tokens: Tuple[int, ...]) -> Tuple[List[int], int]:
        old_state = self._contexts.get(context_id)
        old_block_ids = list(old_state.block_ids) if old_state else []

        matched_block_ids, matched_len = self._find_longest_sealed_prefix(tokens)
        new_block_ids = list(matched_block_ids)
        created_block_ids: List[int] = []
        incref_applied: List[int] = []

        try:
            # First, acquire refs to reused blocks.
            for bid in matched_block_ids:
                self._incref_block(bid)
                incref_applied.append(bid)

            parent_id = new_block_ids[-1] if new_block_ids else None
            cursor = matched_len
            current_prefix = tuple(tokens[:matched_len])
            while cursor < len(tokens):
                chunk = tuple(tokens[cursor : cursor + self.block_size])
                sealed = len(chunk) == self.block_size
                block_id = self._create_block(parent_id, chunk, sealed, current_prefix)
                created_block_ids.append(block_id)
                incref_applied.append(block_id)
                new_block_ids.append(block_id)
                parent_id = block_id
                current_prefix = current_prefix + chunk
                cursor += len(chunk)

            # Commit context first, then release old refs.
            self._contexts[context_id] = ContextState(
                block_ids=new_block_ids,
                tokens=tokens,
                updated_at=time.time(),
            )
            self._decref_chain(old_block_ids)
            self._evict_if_needed()
            return new_block_ids, matched_len
        except Exception:
            # Rollback ref changes and newly created blocks.
            self._safe_rollback(incref_applied, created_block_ids)
            # Keep old state untouched.
            if old_state is None:
                self._contexts.pop(context_id, None)
            else:
                self._contexts[context_id] = old_state
            raise

    def _safe_rollback(self, incref_applied: List[int], created_block_ids: List[int]) -> None:
        # Rollback refs idempotently.
        seen = set()
        for bid in reversed(incref_applied):
            if bid in seen:
                continue
            seen.add(bid)
            block = self._blocks.get(bid)
            if not block:
                continue
            if block.ref_count > 0:
                block.ref_count -= 1
                block.last_access = time.time()
        # Remove newly created zero-ref blocks.
        for bid in created_block_ids:
            block = self._blocks.get(bid)
            if block and block.ref_count == 0:
                self._remove_block(bid)

    def _find_longest_sealed_prefix(self, tokens: Tuple[int, ...]) -> Tuple[List[int], int]:
        matched_block_ids: List[int] = []
        matched_len = 0
        parent_id: Optional[int] = None
        cursor = 0
        prefix: Tuple[int, ...] = ()

        while cursor + self.block_size <= len(tokens):
            chunk = tuple(tokens[cursor : cursor + self.block_size])
            prefix = prefix + chunk
            indexed = self._prefix_index.get(prefix)
            if not indexed:
                break
            bid, generation = indexed
            block = self._blocks.get(bid)
            if (
                block is None
                or block.generation != generation
                or not block.sealed
                or block.parent_id != parent_id
                or block.tokens != chunk
            ):
                break
            matched_block_ids.append(bid)
            matched_len += self.block_size
            parent_id = bid
            cursor += self.block_size
        return matched_block_ids, matched_len

    def _create_block(
        self,
        parent_id: Optional[int],
        tokens: Tuple[int, ...],
        sealed: bool,
        current_prefix: Tuple[int, ...],
    ) -> int:
        block_id = self._next_block_id
        self._next_block_id += 1
        generation = 1
        prefix_key = current_prefix + tokens if sealed else None
        block = KVBlock(
            block_id=block_id,
            generation=generation,
            parent_id=parent_id,
            tokens=tokens,
            sealed=sealed,
            ref_count=1,
            last_access=time.time(),
            prefix_key=prefix_key,
        )
        self._blocks[block_id] = block
        self._total_bytes += block.size_bytes
        if sealed and prefix_key is not None:
            self._prefix_index[prefix_key] = (block_id, generation)
        return block_id

    def _incref_block(self, block_id: int) -> None:
        block = self._blocks.get(block_id)
        if not block:
            raise RuntimeError(f"missing block {block_id}")
        block.ref_count += 1
        block.last_access = time.time()

    def _decref_chain(self, block_ids: List[int]) -> None:
        # Idempotent-ish: never below zero.
        for bid in block_ids:
            block = self._blocks.get(bid)
            if not block:
                continue
            if block.ref_count > 0:
                block.ref_count -= 1
                block.last_access = time.time()

    def _evict_if_needed(self) -> None:
        while len(self._blocks) > self.max_blocks or self._total_bytes > self.max_bytes:
            evict_candidates = [b for b in self._blocks.values() if b.ref_count == 0]
            if not evict_candidates:
                break
            victim = min(evict_candidates, key=lambda b: b.last_access)
            self._remove_block(victim.block_id)

    def _remove_block(self, block_id: int) -> None:
        block = self._blocks.pop(block_id, None)
        if not block:
            return
        self._total_bytes = max(0, self._total_bytes - block.size_bytes)
        if block.prefix_key is not None:
            indexed = self._prefix_index.get(block.prefix_key)
            if indexed and indexed[0] == block_id:
                self._prefix_index.pop(block.prefix_key, None)

    def memory_pressure(self) -> float:
        """返回 KV 内存压力值 (0.0~1.0)"""
        with self._lock:
            block_ratio = len(self._blocks) / self.max_blocks if self.max_blocks > 0 else 0.0
            byte_ratio = self._total_bytes / self.max_bytes if self.max_bytes > 0 else 0.0
            return max(block_ratio, byte_ratio)

    def snapshot_stats(self) -> Dict[str, float]:
        """Return lightweight stats for verification and debugging."""
        with self._lock:
            zero_ref_blocks = sum(1 for b in self._blocks.values() if b.ref_count == 0)
            shared_blocks = sum(1 for b in self._blocks.values() if b.ref_count > 1)
            total_refs = sum(b.ref_count for b in self._blocks.values())
            hit_rate = (
                float(self._prefix_hit_count) / float(self._acquire_count)
                if self._acquire_count > 0
                else 0.0
            )
            avg_matched_tokens = (
                float(self._matched_tokens_total) / float(self._acquire_count)
                if self._acquire_count > 0
                else 0.0
            )
            return {
                "contexts": float(len(self._contexts)),
                "blocks": float(len(self._blocks)),
                "prefix_entries": float(len(self._prefix_index)),
                "total_bytes": float(self._total_bytes),
                "zero_ref_blocks": float(zero_ref_blocks),
                "shared_blocks": float(shared_blocks),
                "total_refs": float(total_refs),
                "acquire_count": float(self._acquire_count),
                "prefix_hit_count": float(self._prefix_hit_count),
                "prefix_hit_rate": hit_rate,
                "avg_matched_tokens": avg_matched_tokens,
            }

    def query_prefix_len(self, tokens: Sequence[int]) -> int:
        """查询前缀命中长度（只读，不修改状态）

        调度器可以用此方法查询某个 token 序列在当前池中的命中情况，
        用于做 KV 感知的路由决策。

        Args:
            tokens: 待查询的 token 序列

        Returns:
            命中的前缀长度（token 数量），0 表示无命中
        """
        token_tuple = tuple(int(t) for t in tokens)
        with self._lock:
            _, matched_len = self._find_longest_sealed_prefix(token_tuple)
            return matched_len

    def debug_context(self, context_id: str) -> Optional[Dict[str, object]]:
        """Return context chain snapshot for tests and diagnostics."""
        with self._lock:
            state = self._contexts.get(context_id)
            if state is None:
                return None
            return {
                "context_id": context_id,
                "tokens": list(state.tokens),
                "block_ids": list(state.block_ids),
                "updated_at": state.updated_at,
            }
