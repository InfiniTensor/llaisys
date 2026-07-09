"""
LLAISYS Chat Server — OpenAI-compatible /v1/chat/completions 接口

用法:
    python chat_server.py --model /path/to/model [--device cpu|nvidia] [--port 8000]

流式调用示例:
    curl http://localhost:8000/v1/chat/completions \\
         -H "Content-Type: application/json" \\
         -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
"""

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import queue
import re
import threading
import time
import uuid
import sys
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Iterator, Set, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic 模型 (OpenAI schema 子集)
# ─────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llaisys"
    messages: List[Message]
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    top_k: int = Field(default=30, ge=1)
    max_tokens: int = Field(default=2048, ge=1)
    stream: bool = False
    # 扩展字段：会话 ID，用于 KV-Cache 前缀复用
    session_id: Optional[str] = "default"
    # 思考预算：<think> 块最多可生成的字符数（0 = 不限制）
    thinking_budget: int = Field(default=800, ge=0)



# ─────────────────────────────────────────────────────────────────────────────
# 用户会话：独立 KV-Cache + 前缀复用逻辑
# ─────────────────────────────────────────────────────────────────────────────

class UserSession:
    """
    轻量级会话标识，不持有 KV-Cache（由 KVCachePool 统一管理）。
    职责：分词 + asyncio.Lock 确保同一 session_id 的请求串行提交给调度器。
    """

    def __init__(self, session_id: str, tokenizer):
        self.session_id = session_id
        self.tokenizer = tokenizer
        self.lock = asyncio.Lock()

    # 默认系统提示：鼓励简洁、避免无限罗列
    _DEFAULT_SYSTEM = (
        "你是一个有帮助的AI助手。请简洁清晰地回答问题，"
        "避免过度展开、无限罗列或重复相似内容。"
        "如需思考，请控制思考长度，直接给出核心答案。"
    )

    def _tokenize(self, messages: List[Message]) -> List[int]:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        # 若没有 system 消息，自动插入默认提示词
        if not msgs or msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": self._DEFAULT_SYSTEM})
        try:
            prompt = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=True
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
        return self.tokenizer.encode(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# KV-Cache Pool：块哈希前缀匹配 + LRU 淘汰
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_SIZE: int = 16          # 每块 token 数（只缓存/匹配「完整 block」）
_ROOT_HASH: bytes = b'\x00' * 8  # 哈希链起始节点


def _block_hash(parent_hash: bytes, block_tokens: List[int], extra: bytes = b"") -> bytes:
    """
    计算单个 KV block 的哈希 key:
        key = H(parent_hash || len(block_tokens) || block_tokens || extra)

    带入 parent_hash 保证不同前缀下相同 block token 不会误命中:
        Prompt A: [X Y][P Q]   vs   Prompt B: [M N][P Q]
        第二块 token 相同，但 parent_hash 不同 → key 不同 → 不会错复用
    """
    h = hashlib.sha256()
    h.update(parent_hash)
    h.update(len(block_tokens).to_bytes(4, "little"))
    for t in block_tokens:
        h.update(t.to_bytes(8, "little"))
    h.update(extra)
    return h.digest()[:8]   # 64-bit，碰撞概率 < 2^{-64}


@dataclass
class KVCacheEntry:
    """
    池中的一个物理 KV 状态条目，封装一个 Qwen2Session。

    Fields:
        entry_id      : 池内唯一 ID
        model_session : Qwen2Session（持有 GPU/CPU KV tensor，shape {maxseq, nkvh, dh}）
        cached_tokens : model_session KV buffer 中已正确计算的完整 token 序列
        block_hashes  : 对应 cached_tokens 的完整 block 哈希链
                        block_hashes[i] = H(block_hashes[i-1], cached_tokens[i*BS:(i+1)*BS])
        ref_cnt       : 借出引用计数（> 0 时不可淘汰，不可被他人借用）
        last_access   : 最近使用时刻（LRU 依据，单调时钟）
        owner_sid     : 最近持有该 entry 的 session_id（同 session 优先复用）
    """
    entry_id: int
    model_session: object
    cached_tokens: List[int] = field(default_factory=list)
    block_hashes: List[bytes] = field(default_factory=list)
    ref_cnt: int = 0
    last_access: float = field(default_factory=time.monotonic)
    owner_sid: str = ""


class KVCachePool:
    """
    跨会话 KV-Cache 前缀匹配池（仿 vLLM block cache 思路）。

    核心数据结构:
        _entries     : Dict[entry_id, KVCacheEntry]  — 全部 entry
        _cache_index : Dict[block_hash, entry_id]    — 只索引完整 block 的末尾 hash
                       key = H(parent_hash, block_tokens)，覆盖整条前缀链
        _free_lru    : OrderedDict[entry_id, None]   — 空闲 entry 的 LRU 链
                       末尾 = 最近使用，头部 = 最旧（淘汰头部）

    前缀匹配流程（borrow）:
        1. 把 prompt_tokens 按 BLOCK_SIZE 切成完整块
        2. 从 _ROOT_HASH 开始逐块计算 block_hash，查询 _cache_index
        3. 找到最长连续命中前缀 → 对应 entry 的 KV data [0, matched_pos) 完全有效
        4. 借出：ref_cnt += 1，从 _free_lru 移除
        5. 调用方 set model_session.cache_pos = matched_pos，只对 suffix 做 prefill

    写回流程（release）:
        1. ref_cnt -= 1
        2. 更新 cached_tokens = prompt_tokens + generated_tokens
        3. 为新完整 block 计算 hash，若 _cache_index 无此 hash 或旧指针已失效则写入
           （只用 token 数更多的 entry 覆盖，保证 index 始终指向最深缓存）
        4. entry 加入 _free_lru 末尾
    """

    def __init__(self, model, max_entries: int = 32):
        self._model = model
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: Dict[int, KVCacheEntry] = {}
        self._cache_index: Dict[bytes, int] = {}         # block_hash → entry_id
        self._free_lru: OrderedDict = OrderedDict()      # entry_id → None (LRU)
        self._next_id: int = 0

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def borrow(
        self,
        prompt_tokens: List[int],
        owner_sid: str = "",
        extra: bytes = b"",
    ) -> Tuple[KVCacheEntry, int]:
        """
        查找并借出 prompt_tokens 的最长前缀命中 entry，返回 (entry, matched_pos)。

        matched_pos : 命中的 token 数（对齐到 BLOCK_SIZE，0 = 无命中）
        entry       : 已借出（ref_cnt += 1）；model_session.cache_pos 已设为 matched_pos

        若无命中：创建新 entry（超出上限时 LRU 淘汰最旧空闲 entry）。
        """
        with self._lock:
            best_entry, best_pos = self._find_best_prefix(
                prompt_tokens, owner_sid, extra
            )
            if best_entry is not None:
                # 命中：rewind session 到前缀边界
                best_entry.model_session.cache_pos = best_pos
                best_entry.ref_cnt += 1
                best_entry.last_access = time.monotonic()
                best_entry.owner_sid = owner_sid
                self._free_lru.pop(best_entry.entry_id, None)
                return best_entry, best_pos
            else:
                # 未命中：分配新 entry（空 session，cache_pos = 0）
                entry = self._alloc_entry()
                entry.ref_cnt = 1
                entry.last_access = time.monotonic()
                entry.owner_sid = owner_sid
                return entry, 0

    def release(
        self,
        entry: KVCacheEntry,
        prompt_tokens: List[int],
        generated_tokens: List[int],
        extra: bytes = b"",
    ) -> None:
        """
        归还借出的 entry，把完整 block 发布到 _cache_index，然后加入 _free_lru。
        cached_tokens 更新为 prompt_tokens + generated_tokens。
        只有「完整 block」会被索引（末尾不足 BLOCK_SIZE 的 token 不缓存）。
        """
        with self._lock:
            # 关键修复：Qwen2Session 的 cache_pos 只统计“作为输入喂给模型”的 token。
            # 生成流程里最后一个已输出 token 通常尚未作为下一步输入写入 KV，
            # 因此不能盲目把 prompt + generated 全部当成可复用 KV。
            full_tokens = list(prompt_tokens) + list(generated_tokens)
            real_cached = int(getattr(entry.model_session, "cache_pos", len(full_tokens)))
            if real_cached < 0:
                real_cached = 0
            if real_cached > len(full_tokens):
                real_cached = len(full_tokens)
            entry.cached_tokens = full_tokens[:real_cached]
            self._publish_blocks(entry, extra)
            entry.ref_cnt = max(0, entry.ref_cnt - 1)
            entry.last_access = time.monotonic()
            if entry.ref_cnt == 0:
                self._free_lru[entry.entry_id] = None
                self._free_lru.move_to_end(entry.entry_id)

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_entries": len(self._entries),
                "free_entries": len(self._free_lru),
                "indexed_blocks": len(self._cache_index),
            }

    def _compute_block_hashes(
        self, tokens: List[int], extra: bytes
    ) -> List[bytes]:
        """返回 tokens 所有完整 block 的哈希链，长度 = len(tokens) // BLOCK_SIZE。"""
        hashes: List[bytes] = []
        parent = _ROOT_HASH
        n_full = len(tokens) // BLOCK_SIZE
        for i in range(n_full):
            bt = tokens[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
            bh = _block_hash(parent, bt, extra)
            hashes.append(bh)
            parent = bh
        return hashes

    def _find_best_prefix(
        self,
        prompt_tokens: List[int],
        owner_sid: str,
        extra: bytes,
    ) -> Tuple[Optional[KVCacheEntry], int]:
        """
        查找最长前缀命中（持锁内调用）。

        逐块计算 block_hash，在 _cache_index 中查找：
          - 命中且 entry 空闲（ref_cnt == 0）→ 更新 best_entry / best_pos
          - 未命中 / entry 被占用             → 停止（后续 hash 也不会命中）
        同等匹配长度时，优先 owner_sid 相同的 entry（多轮对话偏好）。
        """
        n_full = len(prompt_tokens) // BLOCK_SIZE
        if n_full == 0:
            return None, 0

        parent = _ROOT_HASH
        best_entry: Optional[KVCacheEntry] = None
        best_pos: int = 0

        for i in range(n_full):
            bt = prompt_tokens[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
            bh = _block_hash(parent, bt, extra)

            eid = self._cache_index.get(bh)
            if eid is None:
                break                                    # 链断裂，后续不会命中

            entry = self._entries.get(eid)
            if entry is None or entry.ref_cnt > 0:
                break                                    # 无效或被占用

            pos = (i + 1) * BLOCK_SIZE
            # 防御性校验（防止极低概率哈希碰撞）
            if (len(entry.cached_tokens) < pos or
                    entry.cached_tokens[i * BLOCK_SIZE : pos] != bt):
                break
            # 关键修复：必须确保底层 Qwen2Session 的真实 KV 至少覆盖到 pos。
            # 否则会把“逻辑上记录了 token，但 KV 实际未写入”的状态误判为命中，
            # 导致后续解码异常（如提前 EOS、输出中断）。
            real_cache_pos = int(getattr(entry.model_session, "cache_pos", 0))
            if real_cache_pos < pos:
                break

            if (pos > best_pos or
                    (pos == best_pos and entry.owner_sid == owner_sid
                     and (best_entry is None or best_entry.owner_sid != owner_sid))):
                best_entry = entry
                best_pos = pos

            parent = bh

        return best_entry, best_pos

    def _publish_blocks(self, entry: KVCacheEntry, extra: bytes) -> None:
        """
        将 entry.cached_tokens 的完整 block 写入 _cache_index（持锁内调用）。
        若某 block_hash 已被其他 entry 占用，仅当新 entry 的 cached_tokens
        更长（更深缓存）时才覆盖，保证 index 始终指向最长可复用的 entry。
        """
        hashes = self._compute_block_hashes(entry.cached_tokens, extra)
        entry.block_hashes = hashes
        for bh in hashes:
            existing_eid = self._cache_index.get(bh)
            if existing_eid is None or existing_eid not in self._entries:
                self._cache_index[bh] = entry.entry_id
            else:
                existing = self._entries[existing_eid]
                if len(entry.cached_tokens) > len(existing.cached_tokens):
                    self._cache_index[bh] = entry.entry_id  # 更深缓存优先

    def _alloc_entry(self) -> KVCacheEntry:
        """分配新 entry。超出上限时，LRU 淘汰最旧空闲 entry（持锁内调用）。"""
        if len(self._entries) >= self._max_entries and self._free_lru:
            self._evict_lru()
        eid = self._next_id
        self._next_id += 1
        model_session = self._model.create_session()
        entry = KVCacheEntry(entry_id=eid, model_session=model_session)
        self._entries[eid] = entry
        return entry

    def _evict_lru(self) -> None:
        """淘汰 _free_lru 头部（最旧）的空闲 entry（持锁内调用）。"""
        if not self._free_lru:
            return
        eid, _ = self._free_lru.popitem(last=False)    # 弹出最旧（头部）
        entry = self._entries.pop(eid, None)
        if entry is None:
            return
        # 从 _cache_index 中撤销该 entry 的全部 block hash
        for bh in entry.block_hashes:
            if self._cache_index.get(bh) == eid:
                del self._cache_index[bh]
        # Qwen2Session 由 __del__ 自动调用 llaisysQwen2SessionDestroy 释放


# ─────────────────────────────────────────────────────────────────────────────
# 连续批处理调度器：请求队列 + 独立循环线程
# ─────────────────────────────────────────────────────────────────────────────

# 终止生成：session_id → bool，由 /v1/sessions/{sid}/abort 端点设置，_process_token 消费后清除
_session_abort: Dict[str, bool] = {}
_session_abort_mu = threading.Lock()

# 匹配 special token 变体：<|xxx|> / <｜xxx｜> / < | xxx | >（允许空格）
# 使用 [^\n|｜<>] 代替 [\w._-]，以匹配 ▁（U+2581，SentencePiece 词边界符）等非 ASCII 字符
_SPECIAL_FILTER = re.compile(
    r"<\s*[|｜]\s*[^\n|｜<>]{2,}\s*[|｜]\s*>",
    re.UNICODE,
)

# 文本级 EOS 标记检测：模型有时把 EOS 当文本输出而非 token ID
# ▁ = U+2581（LOWER ONE EIGHTH BLOCK），Qwen EOS token 中使用，不是普通下划线
_EOS_TEXT_RE = re.compile(
    r"<\s*[|｜]\s*"
    r"(?:end[▁_\-\s]*of[▁_\-\s]*(?:sentence|text|turn)"
    r"|endoftext"
    r"|im[▁_\-\s]*end"
    r"|eot[▁_\-\s]*id"
    r")\s*[|｜]\s*>",
    re.IGNORECASE,
)

# 流式 hold-back：检测文本尾部可能是特殊 token 的不完整前缀
# [^\n>]* 可匹配含 ▁ 的任意字符，避免 hold-back 失效
_PARTIAL_SPECIAL_TAIL = re.compile(r"<(?:\s{0,3}[|｜][^\n>]*)?$")


def _sanitize_generated_text(text: str) -> str:
        """
        服务端统一文本清洗：
            1) 去掉 <|...|> / <｜...｜> / < | ... | > 特殊 token
            2) 去掉 U+FFFD（常见乱码替换符）
            3) 去掉不可见控制字符（保留换行/制表）
        """
        s = _SPECIAL_FILTER.sub("", text)
        s = s.replace("\ufffd", "")
        s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
        return s


@dataclass
class PendingRequest:
    """HTTP 层提交给调度器的等待请求。"""
    req_id: str
    request: "ChatCompletionRequest"
    session_id: str
    prompt_tokens: List[int]
    result_queue: asyncio.Queue     # ("delta", str) | ("done", None) | ("error", str)
    loop: asyncio.AbstractEventLoop
    tokenizer: object


class ActiveRequest:
    """
    调度器正在迭代解码的单个请求。

    持有借出的 KVCacheEntry，每次 decode_step() 推进一个 token，
    生成结束后通过调度器归还 entry 到 KVCachePool。
    """

    def __init__(
        self,
        pending: PendingRequest,
        entry: KVCacheEntry,
        prompt_tokens: List[int],
        first_token: int,
        end_token: int,
    ):
        self.pending = pending
        self.entry = entry
        self.prompt_tokens = prompt_tokens
        self.end_token = end_token
        self.generated: List[int] = []
        self.accumulated_ids: List[int] = []
        self.text_so_far: str = ""
        self._text_emitted: str = ""          # 已发送给客户端的文本（思考截断后差量基线）
        self.done: bool = False
        self._step: int = 0
        # 思考预算控制
        self._thinking_budget: int = getattr(pending.request, 'thinking_budget', 800)
        self._think_suppressed: bool = False  # 是否正在抑制过长的思考内容
        self._process_token(first_token)

    # ── 文本级多尺度重复检测 ──────────────────────────────────────────────────

    def _check_repetition(self) -> bool:
        """
        文本级多尺度重复检测。

        第一层：短周期连续重复检测（15-200 字符的模式连续出现 ≥3 次）。
                适用于模型反复输出"列表的索引和切片"等短循环。
        第二层：大块子串检测（120-500 字符在前文中出现过）。
                适用于模型在更长尺度上的回环。
        """
        text = self.text_so_far
        n = len(text)
        if n < 60:
            return False

        # ── 第一层：短周期连续重复（3 连击） ──────────────────────────────
        tail = text[-600:] if n > 600 else text
        ct = len(tail)
        for plen in range(15, min(ct // 3 + 1, 201)):
            if ct >= 3 * plen:
                pat = tail[-plen:]
                if (tail[-2 * plen : -plen] == pat
                        and tail[-3 * plen : -2 * plen] == pat):
                    return True

        # ── 第二层：大块子串检测 ──────────────────────────────────────────
        if n < 240:
            return False
        upper = min(n // 2, 500) + 1
        for sz in range(120, upper, 40):
            if text[-sz:] in text[:-sz]:
                return True
        return False

    # ── token 处理 ────────────────────────────────────────────────────────────

    def _process_token(self, tok: int) -> None:
        req = self.pending.request
        self.generated.append(tok)
        self.accumulated_ids.append(tok)
        self._step += 1

        # ── 中止检测（非阻塞，每 token 检查一次）────────────────────────────
        if _session_abort.get(self.pending.session_id, False):
            with _session_abort_mu:
                _session_abort.pop(self.pending.session_id, None)
            self.done = True
            asyncio.run_coroutine_threadsafe(
                self.pending.result_queue.put(("done", None)),
                self.pending.loop,
            )
            return

        # 全量 decode（含特殊 token 文本），用于 EOS 文本检测
        raw_text = self.pending.tokenizer.decode(
            self.accumulated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        # ── 文本级 EOS 检测（在 sanitize 之前） ──────────────────────────────
        eos_match = _EOS_TEXT_RE.search(raw_text)
        if eos_match:
            final_text = _sanitize_generated_text(raw_text[:eos_match.start()])
            delta = final_text[len(self._text_emitted):]
            self._text_emitted = final_text
            self.text_so_far = final_text
            if delta and not self._think_suppressed:
                asyncio.run_coroutine_threadsafe(
                    self.pending.result_queue.put(("delta", delta)),
                    self.pending.loop,
                )
            self.done = True
            asyncio.run_coroutine_threadsafe(
                self.pending.result_queue.put(("done", None)),
                self.pending.loop,
            )
            return

        new_text = _sanitize_generated_text(raw_text)
        self.text_so_far = new_text

        # ── 重复循环检测（在发送 delta 之前）────────────────────────────────
        if self._check_repetition():
            self.done = True
            asyncio.run_coroutine_threadsafe(
                self.pending.result_queue.put(("done", None)),
                self.pending.loop,
            )
            return

        # ── 思考预算控制 ──────────────────────────────────────────────────────
        if not self._think_suppressed:
            t_open = new_text.find('<think>')
            if (self._thinking_budget > 0 and t_open >= 0
                    and new_text.find('</think>', t_open) < 0):
                # 正处于未关闭的 think 块内
                think_len = len(new_text) - t_open - 7
                if think_len > self._thinking_budget:
                    # 思考超出预算：注入 </think> 截断，进入抑制模式
                    asyncio.run_coroutine_threadsafe(
                        self.pending.result_queue.put(("delta", "\n</think>\n\n")),
                        self.pending.loop,
                    )
                    self._think_suppressed = True
                    self._text_emitted = new_text
                    if tok == self.end_token or self._step >= req.max_tokens:
                        self.done = True
                        asyncio.run_coroutine_threadsafe(
                            self.pending.result_queue.put(("done", None)),
                            self.pending.loop,
                        )
                    return

            # 正常路径：计算并发送差量（含 hold-back 机制）
            is_finishing = (tok == self.end_token or self._step >= req.max_tokens)
            # 如果不是最后一步，检查尾部是否有未闭合的 <|... 部分特殊 token
            if is_finishing:
                safe_end = len(new_text)
            else:
                m = _PARTIAL_SPECIAL_TAIL.search(new_text)
                safe_end = m.start() if m else len(new_text)
            delta = new_text[len(self._text_emitted):safe_end]
            self._text_emitted = new_text[:safe_end]
            if delta:
                asyncio.run_coroutine_threadsafe(
                    self.pending.result_queue.put(("delta", delta)),
                    self.pending.loop,
                )
        else:
            # 思考抑制中：等待模型自然关闭 </think>
            t_open = new_text.find('<think>')
            if t_open >= 0 and new_text.find('</think>', t_open) >= 0:
                # 模型已关闭思考块：恢复发送
                t_close = new_text.find('</think>', t_open)
                self._think_suppressed = False
                after_close = new_text[t_close + 8:]   # 8 = len('</think>')
                self._text_emitted = new_text
                if after_close:
                    asyncio.run_coroutine_threadsafe(
                        self.pending.result_queue.put(("delta", after_close)),
                        self.pending.loop,
                    )
            else:
                # 仍在等待，推进基线指针但不发送
                self._text_emitted = new_text

        if tok == self.end_token or self._step >= req.max_tokens:
            self.done = True
            asyncio.run_coroutine_threadsafe(
                self.pending.result_queue.put(("done", None)),
                self.pending.loop,
            )

    def decode_step(self) -> None:
        """推进一步解码（线程安全：各 entry 的 model_session 独立，可并发）。"""
        if self.done:
            return
        req = self.pending.request
        try:
            next_tok = self.entry.model_session._infer_sample(
                [self.generated[-1]],
                req.temperature, req.top_k, req.top_p,
            )
            self._process_token(next_tok)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                self.pending.result_queue.put(("error", str(exc))),
                self.pending.loop,
            )
            self.done = True


class ContinuousBatchScheduler:
    """
    请求池 + 单一循环线程实现的迭代级连续批处理调度器。

    数据流:
        HTTP → pending_queue (queue.Queue, 线程安全)
                    ↓
             调度主循环（独立 daemon 线程，永续运行）
             ┌────────────────────────────────────────────────────┐
             │ Phase 1 – PREFILL（串行，避免多路同时争 GPU）       │
             │   新请求 → KVCachePool.borrow(prompt)              │
             │   → 找最长前缀命中 → 只对 suffix 做 prefill       │
             │   → 首 token 采样 → 构造 ActiveRequest            │
             │                                                    │
             │ Phase 2 – DECODE（线程池并发）                     │
             │   每个 ActiveRequest.decode_step() 在独立线程执行  │
             │   各 entry 持有独立 KV-Cache，互不干扰             │
             │                                                    │
             │ Phase 3 – CLEANUP                                  │
             │   完成请求 → KVCachePool.release()                 │
             │   → 更新 cached_tokens → 发布新 block hash        │
             │   → entry 加入 free_lru，可被后续请求复用          │
             └────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        kv_pool: KVCachePool,
        max_batch_size: int = 8,
        max_workers: int = 16,
    ):
        self._pool = kv_pool
        self._max_batch = max_batch_size
        self._pending: queue.Queue = queue.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="llaisys-decode",
        )
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="llaisys-scheduler"
        )
        self._thread.start()

    def submit(self, req: PendingRequest) -> None:
        """将请求加入等待队列（线程安全，可在任意线程调用）。"""
        self._pending.put(req)

    # ── 内部 ──────────────────────────────────────────────────────────────────

    def _prefill(self, pending: PendingRequest) -> Optional[ActiveRequest]:
        """
        在调度线程内串行执行 prefill:
          1. KVCachePool.borrow(prompt) → (entry, matched_pos)
          2. 只对 prompt[matched_pos:] 做 prefill（KV 前缀复用）
          3. 首 token 采样 → 构造 ActiveRequest
        """
        req = pending.request
        prompt = pending.prompt_tokens
        # 新请求开始前，清除该 session 的任何残留 abort 标志
        with _session_abort_mu:
            _session_abort.pop(pending.session_id, None)
        try:
            entry, matched_pos = self._pool.borrow(
                prompt, owner_sid=pending.session_id
            )
            # 只 feed prefix 之后的 suffix token
            suffix = prompt[matched_pos:]
            if not suffix:
                # 完整命中：回退一步，重新 feed 最后一个 token 以获取首生成 token
                entry.model_session.cache_pos = max(0, matched_pos - 1)
                suffix = [prompt[-1]] if prompt else []
            if not suffix:
                self._pool.release(entry, prompt, [])
                asyncio.run_coroutine_threadsafe(
                    pending.result_queue.put(("done", None)), pending.loop
                )
                return None
            first_tok = entry.model_session._infer_sample(
                suffix, req.temperature, req.top_k, req.top_p
            )
            end_tok = entry.model_session._meta.end_token
            ar = ActiveRequest(pending, entry, prompt, first_tok, end_tok)
            if ar.done:
                self._release(ar)
                return None
            return ar
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                pending.result_queue.put(("error", str(exc))), pending.loop
            )
            return None

    def _release(self, ar: ActiveRequest) -> None:
        """请求完成，归还 entry 到 KVCachePool（更新 cached_tokens + 发布新 block）。"""
        self._pool.release(ar.entry, ar.prompt_tokens, ar.generated)

    def _loop(self) -> None:
        """
        调度主循环（独立 daemon 线程，永续运行）。

        active_sids : 当前活跃 session ID 集合，保证同一 session 同时至多
                      一个活跃请求（防止同一 KV 序列被并发写入）。
        requeue_buf : 因 session 冲突暂缓的请求，当前轮结束后放回队列。
        """
        active: List[ActiveRequest] = []
        active_sids: Set[str] = set()
        requeue_buf: List[PendingRequest] = []

        while True:
            # ── Phase 1: 接受新请求（prefill 串行）────────────────────────────
            slots = self._max_batch - len(active)
            while slots > 0:
                try:
                    pending = self._pending.get_nowait()
                    sid = pending.session_id
                    if sid in active_sids:
                        requeue_buf.append(pending)  # 同 session 已有活跃请求，暂缓
                    else:
                        active_sids.add(sid)
                        ar = self._prefill(pending)
                        if ar is not None:
                            active.append(ar)
                        else:
                            active_sids.discard(sid)
                        slots -= 1
                except queue.Empty:
                    break

            # 暂缓请求放回队列（下一轮调度）
            for p in requeue_buf:
                self._pending.put(p)
            requeue_buf.clear()

            # ── 无活跃请求时阻塞等待新请求 ────────────────────────────────────
            if not active:
                try:
                    pending = self._pending.get(timeout=0.005)
                    sid = pending.session_id
                    active_sids.add(sid)
                    ar = self._prefill(pending)
                    if ar is not None:
                        active.append(ar)
                    else:
                        active_sids.discard(sid)
                except queue.Empty:
                    continue

            # ── Phase 2: 并发执行 decode step ─────────────────────────────────
            if len(active) == 1:
                active[0].decode_step()   # 单请求直接在调度线程执行，省线程切换
            else:
                # 多请求：提交线程池并行执行（各 entry 持有独立 KV-Cache，互不干扰）
                futs = {
                    self._executor.submit(ar.decode_step): ar
                    for ar in active
                }
                for f in concurrent.futures.as_completed(futs):
                    try:
                        f.result()
                    except Exception:
                        pass  # 错误已在 decode_step 内部回传给 result_queue

            # ── Phase 3: 清理已完成请求，归还 KVCachePool ─────────────────────
            still: List[ActiveRequest] = []
            for ar in active:
                if ar.done:
                    self._release(ar)
                    active_sids.discard(ar.pending.session_id)
                else:
                    still.append(ar)
            active = still


# ─────────────────────────────────────────────────────────────────────────────
# ModelServer：会话管理 + KVCachePool + 调度器
# ─────────────────────────────────────────────────────────────────────────────

class ModelServer:
    """
    多用户推理服务核心。

    - KVCachePool     : 跨 session 的 KV-Cache 前缀匹配池（独立物理 session 对象池）
    - ContinuousBatchScheduler : 请求队列 + 循环线程 + 迭代级批处理
    - _sessions       : session_id → UserSession（轻量标识 + tokenizer）
    """

    def __init__(self, model, tokenizer, pool_size: int = 32, max_batch: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self._sessions: Dict[str, UserSession] = {}
        self._sessions_mu = threading.Lock()
        self.kv_pool = KVCachePool(model, max_entries=pool_size)
        self.scheduler = ContinuousBatchScheduler(
            self.kv_pool,
            max_batch_size=max_batch,
            max_workers=max(16, max_batch * 2),
        )

    def get_or_create_session(self, session_id: str) -> UserSession:
        with self._sessions_mu:
            if session_id not in self._sessions:
                self._sessions[session_id] = UserSession(session_id, self.tokenizer)
            return self._sessions[session_id]

    def delete_session(self, session_id: str):
        with self._sessions_mu:
            self._sessions.pop(session_id, None)

    def clear_session(self, session_id: str):
        """将 session_id 的缓存条目在 KVCachePool 中标记为失效（清零其 cached_tokens）。"""
        with self.kv_pool._lock:
            for entry in list(self.kv_pool._entries.values()):
                if entry.owner_sid == session_id and entry.ref_cnt == 0:
                    # 撤销该 entry 的 block hash
                    for bh in entry.block_hashes:
                        if self.kv_pool._cache_index.get(bh) == entry.entry_id:
                            del self.kv_pool._cache_index[bh]
                    entry.cached_tokens = []
                    entry.block_hashes = []
                    entry.model_session.reset_cache()

    def session_count(self) -> int:
        with self._sessions_mu:
            return len(self._sessions)


# ─────────────────────────────────────────────────────────────────────────────
# Web UI（内嵌 HTML）
# ─────────────────────────────────────────────────────────────────────────────

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLAISYS Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f0f2f5; display: flex; height: 100vh; overflow: hidden; }

  /* 左侧会话栏 */
  #sidebar {
    width: 220px; background: #202123; color: #fff; display: flex;
    flex-direction: column; padding: 12px; flex-shrink: 0;
  }
  #sidebar h2 { font-size: 14px; margin-bottom: 12px; color: #aaa; }
  #new-session-btn {
    background: #343541; border: 1px solid #555; color: #fff;
    border-radius: 6px; padding: 8px 12px; cursor: pointer; margin-bottom: 12px;
    font-size: 13px; width: 100%;
  }
  #new-session-btn:hover { background: #444654; }
  #session-list { flex: 1; overflow-y: auto; }
  .session-item {
    padding: 8px 10px; border-radius: 6px; cursor: pointer; font-size: 13px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 4px; user-select: none;
  }
  .session-item:hover { background: #343541; }
  .session-item.active { background: #343541; border-left: 3px solid #10a37f; }
  .session-item .del-btn {
    opacity: 0; font-size: 16px; cursor: pointer; color: #888;
    background: none; border: none; line-height: 1;
  }
  .session-item:hover .del-btn { opacity: 1; }

  /* 右侧主体 */
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

  /* 标题栏 */
  #header {
    padding: 14px 20px; background: #fff; border-bottom: 1px solid #e5e7eb;
    font-size: 15px; font-weight: 600; color: #1a1a1a;
    display: flex; align-items: center; justify-content: space-between;
  }
  #header span { font-size: 12px; font-weight: 400; color: #888; }

  /* 消息区 */
  #messages { flex: 1; overflow-y: auto; padding: 20px; }
  .msg { display: flex; margin-bottom: 20px; animation: fadeIn .2s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; } }
  .msg.user { justify-content: flex-end; }
  .msg.assistant { justify-content: flex-start; }
  .bubble {
    max-width: 75%; padding: 12px 16px; border-radius: 16px;
    line-height: 1.6; font-size: 14px; white-space: pre-wrap; word-break: break-word;
  }
  .msg.user .bubble { background: #10a37f; color: #fff; border-bottom-right-radius: 4px; }
  .msg.assistant .bubble {
    background: #fff; color: #1a1a1a; border-bottom-left-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
  }
  .msg.user .bubble:hover { cursor: pointer; opacity: .9; }
  .edit-hint { font-size: 11px; color: rgba(255,255,255,.6); margin-top: 4px; text-align: right; }

  /* 输入区 */
  #input-area {
    padding: 16px 20px; background: #fff; border-top: 1px solid #e5e7eb;
    display: flex; gap: 10px; align-items: flex-end;
  }
  #user-input {
    flex: 1; border: 1px solid #d1d5db; border-radius: 10px; padding: 10px 14px;
    font-size: 14px; resize: none; outline: none; max-height: 150px; overflow-y: auto;
    line-height: 1.5; font-family: inherit;
  }
  #user-input:focus { border-color: #10a37f; }
  #send-btn {
    background: #10a37f; color: #fff; border: none; border-radius: 10px;
    padding: 10px 18px; cursor: pointer; font-size: 14px; font-weight: 600;
    transition: background .15s; white-space: nowrap;
  }
  #send-btn:hover:not(:disabled) { background: #0d8a6b; }
  #send-btn:disabled { background: #ccc; cursor: not-allowed; }

  #params { font-size: 12px; color: #888; padding: 6px 20px 0;
            display: flex; gap: 16px; align-items: center; }
  #params label { display: flex; align-items: center; gap: 4px; }
  #params input[type=range] { width: 80px; }
  #params input[type=number] { width: 50px; border: 1px solid #ddd; border-radius: 4px;
                                 padding: 2px 4px; font-size: 12px; }
  .cursor { display: inline-block; width: 2px; height: 1em; background: #555;
            animation: blink .7s step-end infinite; vertical-align: text-bottom; }
  @keyframes blink { 50% { opacity: 0; } }
</style>
</head>
<body>

<div id="sidebar">
  <h2>LLAISYS Chat</h2>
  <button id="new-session-btn" onclick="newSession()">+ 新建对话</button>
  <div id="session-list"></div>
</div>

<div id="main">
  <div id="header">
    <span id="session-title">对话</span>
    <div style="display:flex;gap:8px;align-items:center">
      <button onclick="clearCurrentSession()" id="clear-btn"
        style="font-size:12px;padding:4px 10px;border:1px solid #ddd;border-radius:6px;
               background:#fff;color:#888;cursor:pointer" title="清空当前会话历史">清空</button>
      <span id="model-info">LLAISYS</span>
    </div>
  </div>

  <div id="params">
    <label>温度
      <input type="range" id="temperature" min="0" max="2" step="0.1" value="1"
             oninput="document.getElementById('temp-val').textContent=this.value">
      <span id="temp-val">0.5</span>
    </label>
    <label>Top-K <input type="number" id="top_k" value="30" min="1" max="500"></label>
    <label>Top-P <input type="number" id="top_p" value="0.9" min="0" max="1" step="0.05"></label>
    <label>最大 Token <input type="number" id="max_tokens" value="2048" min="1" max="4096"></label>
    <label>思考上限 <input type="number" id="thinking_budget" value="800" min="0" max="4000"></label>
  </div>

  <div id="messages"></div>

  <div id="input-area">
    <textarea id="user-input" rows="2" placeholder="输入消息，Shift+Enter 发送，Enter 换行…"
              onkeydown="handleKey(event)"></textarea>
    <button id="send-btn" onclick="sendMessage()">发送</button>
    <button id="stop-btn" onclick="stopGenerate()" disabled
      style="background:#fff;color:#dc2626;border:1px solid #fca5a5;border-radius:10px;
             padding:10px 14px;cursor:pointer;font-size:14px;font-weight:500;
             transition:all 0.15s;">停止</button>
  </div>
</div>

<script>
const SERVER = "";
// session_id  { title, messages[], isGenerating, pendingReply, _bubble }
let sessions = {};
let currentSessionId = null;

//  localStorage 
function saveSessions() {
  try {
    const data = {};
    for (const [id, s] of Object.entries(sessions))
      data[id] = { title: s.title, messages: s.messages, isGenerating: false, pendingReply: "" };
    localStorage.setItem("llaisys_sessions", JSON.stringify(data));
  } catch {}
}

//  会话管理 
function genId() { return "s_" + Math.random().toString(36).slice(2, 10); }

function newSession() {
  const id = genId();
  sessions[id] = { title: "新对话", messages: [], isGenerating: false, pendingReply: "", _bubble: null };
  currentSessionId = id;
  renderSidebar();
  renderMessages();
  document.getElementById("send-btn").disabled = false;
  saveSessions();
}

function switchSession(id) {
  if (id === currentSessionId) return;
  currentSessionId = id;
  renderSidebar();
  renderMessages();  // 如该会话正在生成，renderMessages 会重建 _bubble 并显示已生成内容
  document.getElementById("send-btn").disabled = sessions[id]?.isGenerating || false;
  // 不再调用 /clear：每个 session_id 在服务器端有独立 KV-Cache，切换不应重置
}

function deleteSession(id, e) {
  e.stopPropagation();
  fetch(`${SERVER}/v1/sessions/${id}`, { method: "DELETE" }).catch(() => {});
  delete sessions[id];
  if (currentSessionId === id) {
    const ids = Object.keys(sessions);
    if (ids.length) {
      currentSessionId = ids[ids.length - 1];
      renderSidebar();
      renderMessages();
      document.getElementById("send-btn").disabled = sessions[currentSessionId]?.isGenerating || false;
    } else {
      newSession();
      return;
    }
  } else {
    renderSidebar();
  }
  saveSessions();
}

function clearCurrentSession() {
  const sess = sessions[currentSessionId];
  if (!sess || sess.isGenerating) return;
  sess.messages = [];
  sess.title = "新对话";
  sess.pendingReply = "";
  fetch(`${SERVER}/v1/sessions/${currentSessionId}/clear`, { method: "POST" }).catch(() => {});
  renderSidebar();
  renderMessages();
  saveSessions();
}

function renderSidebar() {
  const list = document.getElementById("session-list");
  list.innerHTML = "";
  for (const [id, s] of Object.entries(sessions).reverse()) {
    const el = document.createElement("div");
    el.className = "session-item" + (id === currentSessionId ? " active" : "");
    el.onclick = () => switchSession(id);
    el.innerHTML = `<span>${s.title.slice(0, 20)}</span>
      <button class="del-btn" onclick="deleteSession('${id}', event)"></button>`;
    list.appendChild(el);
  }
  document.getElementById("session-title").textContent =
    sessions[currentSessionId]?.title || "对话";
}

//  消息渲染 
function renderMessages() {
  const container = document.getElementById("messages");
  container.innerHTML = "";
  const sess = sessions[currentSessionId];
  if (!sess) return;
  sess.messages.forEach((m, i) => appendBubble(m.role, m.content, i));
  // 如该会话正在生成，重建流式气泡并更新 _bubble 引用
  // （切换回此会话时可看到已生成的内容）
  if (sess.isGenerating) {
    const b = appendBubble("assistant", sess.pendingReply || "");
    if (!sess.pendingReply) b.innerHTML = '<span class="cursor"></span>';
    sess._bubble = b;
  }
  container.scrollTop = container.scrollHeight;
}

function appendBubble(role, content, idx) {
  const container = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.id = `msg-${idx ?? Date.now()}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  if (role === "user") {
    bubble.title = "点击编辑并重新生成";
    bubble.onclick = () => editMessage(idx);
    const hint = document.createElement("div");
    hint.className = "edit-hint";
    hint.textContent = "点击可编辑";
    div.appendChild(bubble);
    div.appendChild(hint);
  } else {
    div.appendChild(bubble);
  }
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

//  编辑历史消息（重新生成） 
function editMessage(idx) {
  if (sessions[currentSessionId]?.isGenerating) return;
  const session = sessions[currentSessionId];
  const editedContent = prompt("编辑消息：", session.messages[idx].content);
  if (editedContent === null) return;
  session.messages[idx].content = editedContent;
  session.messages = session.messages.slice(0, idx + 1);
  renderMessages();
  doGenerate();
}

//  发送消息 
function handleKey(e) {
  if (e.key === "Enter" && e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
  // 普通 Enter = 换行（走默认行为）
}

function sendMessage() {
  const sess = sessions[currentSessionId];
  if (!sess || sess.isGenerating) return;
  const input = document.getElementById("user-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.style.height = "auto";
  sess.messages.push({ role: "user", content: text });
  if (sess.messages.length === 1) { sess.title = text.slice(0, 24); renderSidebar(); }
  appendBubble("user", text, sess.messages.length - 1);
  saveSessions();
  doGenerate();
}

async function doGenerate() {
  const genSessionId = currentSessionId;
  const session = sessions[genSessionId];
  if (!session || session.isGenerating) return;

  session.isGenerating = true;
  session.pendingReply = "";
  // 仅对当前可见会话禁用发送按钮
  if (currentSessionId === genSessionId) {
    document.getElementById("send-btn").disabled = true;
    document.getElementById("stop-btn").disabled = false;
  }

  const temperature = parseFloat(document.getElementById("temperature").value);
  const top_k = parseInt(document.getElementById("top_k").value);
  const top_p = parseFloat(document.getElementById("top_p").value);
  const max_tokens = parseInt(document.getElementById("max_tokens").value);

  // 创建初始气泡并将引用存入 session
  // （renderMessages 切换会话时会更新 session._bubble 指向新 DOM 元素）
  const bubble = appendBubble("assistant", "");
  bubble.innerHTML = '<span class="cursor"></span>';
  session._bubble = bubble;
  let reply = "";

  try {
    const resp = await fetch(`${SERVER}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "llaisys",
        messages: session.messages,
        temperature, top_k, top_p, max_tokens,
        thinking_budget: parseInt(document.getElementById("thinking_budget")?.value || "800"),
        stream: true,
        session_id: genSessionId,
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", streamDone = false;

    while (!streamDone) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") { streamDone = true; break; }
        try {
          const delta = JSON.parse(payload).choices?.[0]?.delta?.content || "";
          if (!delta) continue;
          reply += delta;
          session.pendingReply = reply;
          // session._bubble 可能被 renderMessages 更新为新 DOM 元素，始终用最新引用
          if (session._bubble && document.body.contains(session._bubble)) {
            session._bubble.textContent = reply;
            document.getElementById("messages").scrollTop = 99999;
          }
        } catch {}
      }
    }
    reader.cancel().catch(() => {});
  } catch (err) {
    reply = reply || `[错误: ${err.message}]`;
    if (session._bubble && document.body.contains(session._bubble))
      session._bubble.textContent = reply;
  } finally {
    session.pendingReply = "";
    session._bubble = null;
    session.isGenerating = false;
    session.messages.push({ role: "assistant", content: reply || "[无回复]" });
    saveSessions();
    // 如果用户当前在此会话，重新渲染消息并恢复按钮
    if (currentSessionId === genSessionId) {
      renderMessages();
      document.getElementById("send-btn").disabled = false;
      document.getElementById("stop-btn").disabled = true;
    }
  }
}

async function stopGenerate() {
  const sess = sessions[currentSessionId];
  if (!sess || !sess.isGenerating) return;
  // 通知服务端中止
  fetch(`${SERVER}/v1/sessions/${currentSessionId}/abort`, { method: "POST" }).catch(() => {});
  // 更新 UI
  const reply = sess.pendingReply || "[已停止]";
  sess.isGenerating = false;
  sess.pendingReply = "";
  if (sess._bubble && document.body.contains(sess._bubble))
    sess._bubble.textContent = reply;
  sess._bubble = null;
  sess.messages.push({ role: "assistant", content: reply });
  saveSessions();
  document.getElementById("send-btn").disabled = false;
  document.getElementById("stop-btn").disabled = true;
  renderMessages();
}

//  初始化 
(function() {
  try {
    const saved = localStorage.getItem("llaisys_sessions");
    if (saved) {
      const data = JSON.parse(saved);
      const ids = Object.keys(data || {});
      if (ids.length > 0) {
        sessions = data;
        for (const s of Object.values(sessions)) {
          s.isGenerating = false;
          s.pendingReply = "";
          s._bubble = null;
        }
        currentSessionId = ids[ids.length - 1];
        renderSidebar();
        renderMessages();
        document.getElementById("send-btn").disabled = false;
        return;
      }
    }
  } catch {}
  newSession();
})();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 应用
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="LLAISYS Chat Server", version="0.2.0")
_server: Optional[ModelServer] = None


def _make_sse_chunk(content: str, chat_id: str, model: str) -> str:
    data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _make_sse_done(chat_id: str, model: str) -> str:
    data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"


@app.get("/", response_class=HTMLResponse)
def index():
    return WEB_UI_HTML


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible /v1/chat/completions。

    所有请求统一提交到 ContinuousBatchScheduler:
      - 分词在 HTTP handler 中完成（纯 CPU，不阻塞事件循环）
      - PendingRequest 入队 → 调度器 prefill（KVCachePool 前缀复用）
      - 调度器 decode 循环推进 → token 通过 result_queue 回传事件循环
      - 流式：SSE 实时推送；非流式：等全部 token 后一次性返回
    """
    if _server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    sid = request.session_id or "default"
    user_session = _server.get_or_create_session(sid)

    loop = asyncio.get_event_loop()
    result_queue: asyncio.Queue = asyncio.Queue(maxsize=512)
    prompt_tokens = user_session._tokenize(request.messages)

    pending = PendingRequest(
        req_id=chat_id,
        request=request,
        session_id=sid,
        prompt_tokens=prompt_tokens,
        result_queue=result_queue,
        loop=loop,
        tokenizer=user_session.tokenizer,
    )
    _server.scheduler.submit(pending)

    if request.stream:
        async def sse_gen():
            while True:
                kind, data = await result_queue.get()
                if kind == "done":
                    yield _make_sse_done(chat_id, request.model)
                    break
                elif kind == "error":
                    yield _make_sse_chunk(f"[Error: {data}]", chat_id, request.model)
                    yield _make_sse_done(chat_id, request.model)
                    break
                elif kind == "delta" and data:
                    yield _make_sse_chunk(data, chat_id, request.model)

        return StreamingResponse(
            sse_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    else:
        parts: List[str] = []
        while True:
            kind, data = await result_queue.get()
            if kind == "done":
                break
            elif kind == "error":
                raise HTTPException(status_code=500, detail=data)
            elif kind == "delta" and data:
                parts.append(data)
        reply = "".join(parts)
        return JSONResponse({
            "id": chat_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": len(prompt_tokens),
                      "completion_tokens": -1, "total_tokens": -1},
        })


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [
        {"id": "llaisys", "object": "model",
         "active_sessions": _server.session_count() if _server else 0}
    ]}


@app.post("/v1/sessions/{session_id}/clear")
def clear_session(session_id: str):
    """清除指定会话的 KV-Cache（会话切换时调用）。"""
    if _server is not None:
        _server.clear_session(session_id)
    return {"status": "ok", "session_id": session_id}


@app.post("/v1/sessions/{session_id}/abort")
def abort_session(session_id: str):
    """中止指定会话的当前生成（最多延迟 1-2 个 token 后停止）。"""
    with _session_abort_mu:
        _session_abort[session_id] = True
    return {"status": "ok", "session_id": session_id}


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str):
    """彻底删除会话并释放其 KV-Cache 内存。"""
    if _server is not None:
        _server.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/v1/sessions")
def list_sessions():
    """查看当前活跃会话数量及 KVCachePool 统计。"""
    count = _server.session_count() if _server else 0
    pool_stats = _server.kv_pool.stats() if _server else {}
    return {"active_sessions": count, "kv_pool": pool_stats}


# ─────────────────────────────────────────────────────────────────────────────
# 启动入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global _server

    parser = argparse.ArgumentParser(
        description="LLAISYS Chat Server (OpenAI-compatible, continuous batching + KV prefix cache)"
    )
    parser.add_argument("--model", required=True, help="模型目录路径")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pool-size", type=int, default=32,
                        help="KVCachePool 最大 entry 数（每个 entry 占一个 Qwen2Session 的 KV 显存）")
    parser.add_argument("--max-batch", type=int, default=8,
                        help="调度器每轮最大并发请求数")
    args = parser.parse_args()

    import llaisys
    from llaisys import DeviceType
    from transformers import AutoTokenizer

    device = DeviceType.NVIDIA if args.device == "nvidia" else DeviceType.CPU

    print(f"[server] 加载 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[server] 加载模型 (device={args.device}): {args.model}")
    model = llaisys.models.Qwen2(args.model, device)
    print("[server] 模型加载完成。")

    _server = ModelServer(model, tokenizer,
                          pool_size=args.pool_size,
                          max_batch=args.max_batch)

    print(f"[server] KVCachePool 已启动: max_entries={args.pool_size}, block_size={BLOCK_SIZE} tokens")
    print(f"[server] 调度器已启动: max_batch={args.max_batch}，前缀命中时只 prefill suffix")
    print(f"[server] 监听 http://{args.host}:{args.port}")
    print(f"[server] Web UI: http://localhost:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
