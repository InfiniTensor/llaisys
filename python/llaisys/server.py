from __future__ import annotations

import argparse
import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import llaisys
from llaisys.interfaces import IInferenceService
from llaisys.kv_cache_pool import KVCachePool
from llaisys.kv_runtime_bridge import KVRuntimeBridge
from llaisys.libllaisys import LlaisysSamplingParams
from llaisys.models import Qwen2
from llaisys.scheduler import InferenceScheduler, SchedulerQueueFullError, TaskTimeoutError


# ---------------------------------------------------------------------------
# Streaming batch data structures
# ---------------------------------------------------------------------------

@dataclass
class BatchSequenceState:
    index: int
    context_id: str
    messages: List[Dict[str, str]]
    prompt_ids: List[int]
    generated_ids: List[int] = field(default_factory=list)
    filtered_text: str = ""
    max_new_tokens: int = 128
    sampling: Dict[str, Any] = field(default_factory=dict)
    sampling_params: Optional[LlaisysSamplingParams] = None
    use_sampling: bool = False
    cancel_event: Optional[threading.Event] = None
    finished: bool = False
    finish_reason: Optional[str] = None


@dataclass
class BatchState:
    sequences: List[BatchSequenceState]
    any_sampling: bool
    eos_token: int


@dataclass
class StepResult:
    seq_index: int
    delta_text: str
    finished: bool
    finish_reason: Optional[str]
    stopped: bool = False
from llaisys.session_manager import SessionManager


def _wrap_completion(
    session_id: str,
    content: str,
    finish_reason: str,
    usage: Dict[str, int],
    stopped: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "id": f"chatcmpl-{session_id}",
        "object": "chat.completion",
        "model": "qwen2",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
        "session_id": session_id,
    }
    if stopped:
        result["stopped"] = True
    return result


def _wrap_chunk(
    session_id: str,
    delta_content: Optional[str],
    finish_reason: Optional[str],
    usage: Optional[Dict[str, int]] = None,
    stopped: bool = False,
) -> Dict[str, Any]:
    delta: Dict[str, str] = {}
    if delta_content is not None:
        delta["content"] = delta_content
    chunk: Dict[str, Any] = {
        "id": f"chatcmpl-{session_id}",
        "object": "chat.completion.chunk",
        "model": "qwen2",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
        "session_id": session_id,
    }
    if usage is not None:
        chunk["usage"] = usage
    if stopped:
        chunk["stopped"] = True
    return chunk


def _wrap_error(message: str, error_type: str = "server_error", code: str = "") -> Dict[str, Any]:
    err: Dict[str, Any] = {"error": {"message": message, "type": error_type}}
    if code:
        err["error"]["code"] = code
    return err


class ChatService(IInferenceService):
    def __init__(
        self,
        model: Qwen2,
        tokenizer: llaisys.Tokenizer,
        model_path: Optional[str] = None,
        enable_kv_runtime_reuse: bool = False,
        block_size: int = 64,
        max_blocks: int = 4096,
        max_bytes: int = 256 * 1024 * 1024,
        model_lock: Optional[threading.RLock] = None,
        kv_pool: Optional[KVCachePool] = None,
        kv_bridge: Optional[KVRuntimeBridge] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._enable_kv_runtime_reuse = bool(enable_kv_runtime_reuse)
        # RLock allows cooperative iterator-level scheduling in continuous-batching mode.
        self._model_lock = model_lock if model_lock is not None else threading.RLock()

        # Delegated components
        self._session_mgr = SessionManager()
        self._kv_bridge = kv_bridge if kv_bridge is not None else KVRuntimeBridge(model, enabled=enable_kv_runtime_reuse)
        self._kv_pool = kv_pool if kv_pool is not None else KVCachePool(
            block_size=block_size,
            max_blocks=max_blocks,
            max_bytes=max_bytes,
        )
        self._active_tokens: List[int] = []

        # Text processing
        self._chat_template_tokenizer = self._init_chat_template_tokenizer(model_path)
        self._filter_tokens = ("<|end_of_sentence|>",)
        self._filter_patterns = [
            re.compile(r"<\s*\|\s*end_of_sentence\s*\|\s*>", re.IGNORECASE),
            re.compile(r"<\s*\|[^>]*\|\s*>"),
            re.compile(r"<\s*[\|\uFF5C][^>]*[\|\uFF5C]\s*>"),
            re.compile(
                r"<\s*[\|\uFF5C]\s*end[\s_\u2581]*of[\s_\u2581]*sentence\s*[\|\uFF5C]\s*>",
                re.IGNORECASE,
            ),
        ]

    @property
    def kv_pool(self) -> KVCachePool:
        """暴露 KVCache 池给调度器查询"""
        return self._kv_pool

    def tokenize_for_routing(self, payload: Dict[str, Any]) -> Optional[List[int]]:
        """为 KV 感知路由进行轻量级 tokenize

        尝试从 payload 构建 prompt 并 tokenize，用于调度器查询 KV 命中。
        失败时返回 None，不影响正常请求处理。

        Args:
            payload: 请求参数

        Returns:
            token ids 列表，或 None（如果无法 tokenize）
        """
        try:
            # 尝试提取 messages
            messages = payload.get("messages")
            prompt_text = payload.get("prompt")
            system_prompt = payload.get("system_prompt")

            if messages is not None:
                if not isinstance(messages, list):
                    return None
                prompt = self._render_prompt(list(messages), str(system_prompt) if system_prompt else None)
            elif prompt_text is not None:
                # 简单 prompt，尝试获取历史
                session_id = str(payload.get("session_id") or "").strip()
                history = self._session_mgr.get_messages(session_id)
                history.append({"role": "user", "content": str(prompt_text)})
                prompt = self._render_prompt(history, str(system_prompt) if system_prompt else None)
            else:
                return None

            return self.tokenizer.encode(prompt)
        except Exception:
            return None

    @staticmethod
    def _init_chat_template_tokenizer(model_path: Optional[str]):
        if not model_path:
            return None
        try:
            from transformers import AutoTokenizer
        except Exception:
            return None
        try:
            return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            return None

    def _postprocess_text(self, text: str) -> str:
        for token in self._filter_tokens:
            text = text.replace(token, "")
        for pattern in self._filter_patterns:
            text = pattern.sub("", text)
        return text

    def request_stop(self, context_id: str) -> bool:
        return self._session_mgr.request_stop(context_id)

    def kv_debug_snapshot(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        snapshot = self._kv_bridge.debug_snapshot(session_id)
        snapshot["kv_pool"] = self._kv_pool.snapshot_stats()
        return snapshot

    def _render_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str]) -> str:
        templated_messages: List[Dict[str, str]] = []
        if system_prompt:
            templated_messages.append({"role": "system", "content": str(system_prompt)})
        templated_messages.extend(messages)

        if self._chat_template_tokenizer is not None:
            try:
                return self._chat_template_tokenizer.apply_chat_template(
                    templated_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                pass
        return Qwen2.build_prompt(
            messages,
            system_prompt=str(system_prompt) if system_prompt else None,
            add_generation_prompt=True,
        )

    def _eos_token(self) -> int:
        eos = getattr(self.model, "_meta", None)
        if eos is None:
            return -1
        end_token = getattr(eos, "end_token", -1)
        return int(getattr(end_token, "value", end_token))

    def _decode_next(
        self,
        token_ids: List[int],
        use_sampling: bool,
        sampling: Dict[str, Any],
    ) -> int:
        top_k = int(sampling.get("top_k", 1))
        top_p = float(sampling.get("top_p", 0.0))
        temperature = float(sampling.get("temperature", 0.0))
        seed = int(sampling.get("seed", 0))
        if use_sampling:
            return int(
                self.model.step_sampling(
                    token_ids,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    seed=seed,
                )
            )
        return int(self.model.step(token_ids))

    def _prefill_next(
        self,
        prompt_ids: List[int],
        use_sampling: bool,
        sampling: Dict[str, Any],
    ) -> int:
        top_k = int(sampling.get("top_k", 1))
        top_p = float(sampling.get("top_p", 0.0))
        temperature = float(sampling.get("temperature", 0.0))
        seed = int(sampling.get("seed", 0))
        if use_sampling:
            return int(
                self.model.prefill_sampling(
                    prompt_ids,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    seed=seed,
                )
            )
        return int(self.model.prefill(prompt_ids))

    def _iter_generate_ids(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        sampling: Dict[str, Any],
        prefix_len: int,
        cancel_event: threading.Event,
    ) -> Iterable[int]:
        mode = str(sampling.get("mode", "")).strip().lower()
        top_k = int(sampling.get("top_k", 1))
        top_p = float(sampling.get("top_p", 0.0))
        temperature = float(sampling.get("temperature", 0.0))
        if mode == "argmax":
            use_sampling = False
        elif mode == "sample":
            use_sampling = True
        else:
            use_sampling = temperature > 0.0 or top_k > 1 or top_p > 0.0

        if cancel_event.is_set():
            return

        can_reuse_active_prefix = (
            self._enable_kv_runtime_reuse
            and prefix_len > 0
            and len(self._active_tokens) == prefix_len
            and self._active_tokens[:prefix_len] == prompt_ids[:prefix_len]
            and len(prompt_ids) > prefix_len
        )
        if can_reuse_active_prefix:
            next_token = self._decode_next(prompt_ids[prefix_len:], use_sampling, sampling)
            self._active_tokens = list(prompt_ids)
        else:
            self.model.reset_kv_cache()
            next_token = self._prefill_next(prompt_ids, use_sampling, sampling)
            self._active_tokens = list(prompt_ids)

        if next_token < 0:
            return

        eos = self._eos_token()
        yield next_token
        self._active_tokens.append(next_token)
        for _ in range(max_new_tokens - 1):
            if cancel_event.is_set():
                break
            if eos >= 0 and next_token == eos:
                break
            next_token = self._decode_next([next_token], use_sampling, sampling)
            if next_token < 0:
                break
            yield next_token
            self._active_tokens.append(next_token)

    def _prepare_request(self, payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]], List[int], Dict[str, Any], int]:
        system_prompt = payload.get("system_prompt")
        # Accept OpenAI's max_tokens as alias; prefer it over max_new_tokens
        if "max_tokens" in payload:
            max_new_tokens = int(payload["max_tokens"])
        else:
            max_new_tokens = int(payload.get("max_new_tokens", 128))
        # model field accepted and ignored
        sampling = {
            "mode": payload.get("sampling"),
            "top_k": payload.get("top_k", 1),
            "top_p": payload.get("top_p", 0.0),
            "temperature": payload.get("temperature", 0.0),
            "seed": payload.get("seed", 0),
        }

        context_id, messages = self._session_mgr.extract_messages(payload)
        prompt = self._render_prompt(messages, str(system_prompt) if system_prompt else None)
        prompt_ids = self.tokenizer.encode(prompt)
        return context_id, messages, prompt_ids, sampling, max_new_tokens

    def generate_packed_non_stream(self, payloads: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Best-effort packed non-stream path (greedy + sampling).

        Current safe scope:
        - non-stream requests only
        - greedy and sampling requests (mixed batches supported)
        - no history-edit branching fields

        When any request uses sampling, the batch is routed through the
        packed-sampling C API.  If that API is unavailable (old DLL), sampling
        requests fall back to ``None`` so the scheduler handles them one by one.
        Pure-greedy batches still use the original fast ``prefill_packed`` path.
        """
        if not payloads:
            return []
        if not hasattr(self.model, "prefill_packed") or not hasattr(self.model, "step_packed"):
            return None

        prepared: List[Tuple[str, List[Dict[str, str]], List[int], Dict[str, Any], int]] = []
        any_sampling = False
        sampling_params_list: List[LlaisysSamplingParams] = []
        for payload in payloads:
            if payload.get("stream", False):
                return None
            # History editing introduces branch semantics; keep packed path conservative for now.
            if payload.get("edit_from_session_id"):
                return None
            try:
                context_id, messages, prompt_ids, sampling, max_new_tokens = self._prepare_request(payload)
            except Exception:
                return None
            if max_new_tokens <= 0:
                return None
            mode = str(sampling.get("mode", "")).strip().lower()
            top_k = int(sampling.get("top_k", 1))
            top_p = float(sampling.get("top_p", 0.0))
            temperature = float(sampling.get("temperature", 0.0))
            seed = int(sampling.get("seed", 0))
            if mode == "argmax":
                use_sampling = False
            elif mode == "sample":
                use_sampling = True
            else:
                use_sampling = temperature > 0.0 or top_k > 1 or top_p > 0.0
            if use_sampling:
                any_sampling = True
            sampling_params_list.append(LlaisysSamplingParams(
                top_k=top_k, top_p=top_p,
                temperature=temperature, seed=seed,
            ))
            prepared.append((context_id, messages, prompt_ids, sampling, max_new_tokens))

        # If any request needs sampling, check for the packed-sampling API.
        # Fall back to None (single-request path) when the new DLL is absent.
        if any_sampling:
            if not hasattr(self.model, "prefill_packed_sampling") or not hasattr(self.model, "step_packed_sampling"):
                return None

        prompts = [it[2] for it in prepared]
        generated_all: List[List[int]] = [[] for _ in prepared]
        last_step_inputs: List[int] = [int(p[-1]) if p else 0 for p in prompts]
        max_new_tokens_list = [int(it[4]) for it in prepared]
        eos = self._eos_token()
        with self._model_lock:
            self.model.reset_kv_cache()
            if any_sampling:
                next_tokens = self.model.prefill_packed_sampling(prompts, sampling_params_list)
            else:
                next_tokens = self.model.prefill_packed(prompts)
            if len(next_tokens) != len(prepared):
                return None
            for i, tok in enumerate(next_tokens):
                t = int(tok)
                if t >= 0:
                    generated_all[i].append(t)
                    last_step_inputs[i] = t
            # Continue decode rounds for unfinished requests (dynamic shrinking).
            while True:
                active_indices: List[int] = []
                decode_inputs: List[List[int]] = []
                active_sp: List[LlaisysSamplingParams] = []
                for i in range(len(generated_all)):
                    gen = generated_all[i]
                    if not gen:
                        continue
                    if len(gen) >= max_new_tokens_list[i]:
                        continue
                    if eos >= 0 and gen[-1] == eos:
                        continue
                    active_indices.append(i)
                    decode_inputs.append([int(last_step_inputs[i])])
                    active_sp.append(sampling_params_list[i])
                if not active_indices:
                    break
                if any_sampling:
                    step_tokens = self.model.step_packed_sampling(decode_inputs, active_sp)
                else:
                    step_tokens = self.model.step_packed(decode_inputs)
                if len(step_tokens) != len(active_indices):
                    return None
                for j, tok in enumerate(step_tokens):
                    ai = active_indices[j]
                    t = int(tok)
                    if t >= 0:
                        generated_all[ai].append(t)
                        last_step_inputs[ai] = t

        out: List[Dict[str, Any]] = []
        for i, (context_id, messages, prompt_ids, _sampling, _max_new_tokens) in enumerate(prepared):
            generated_ids = list(generated_all[i])
            response_text = self._postprocess_text(self.tokenizer.decode(generated_ids))
            messages2 = list(messages)
            messages2.append({"role": "assistant", "content": response_text})
            self._session_mgr.save_messages(context_id, messages2)
            self._session_mgr.clear_stop(context_id)
            usage = {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(prompt_ids) + len(generated_ids),
            }
            hit_limit = len(generated_ids) >= _max_new_tokens
            finish_reason = "length" if hit_limit else "stop"
            out.append(_wrap_completion(context_id, response_text, finish_reason, usage))
        return out

    # Backward-compatible alias used by scheduler tests/mocks.
    def generate_packed_once(self, payloads: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        return self.generate_packed_non_stream(payloads)

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        context_id, messages, prompt_ids, sampling, max_new_tokens = self._prepare_request(payload)
        cancel_event = self._session_mgr.get_cancel_event(context_id)
        self._session_mgr.clear_stop(context_id)

        with self._model_lock:
            acquire = self._kv_pool.acquire_context(context_id, prompt_ids)
            self._kv_bridge.bind_for_request(context_id, prompt_ids, acquire.prefix_len)
            generated_ids: List[int] = []
            try:
                for token_id in self._iter_generate_ids(
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    sampling=sampling,
                    prefix_len=acquire.prefix_len,
                    cancel_event=cancel_event,
                ):
                    generated_ids.append(int(token_id))
                cancelled = cancel_event.is_set()
                if cancelled:
                    self._active_tokens = list(prompt_ids)
                    self._kv_pool.update_context(context_id, prompt_ids)
                else:
                    self._kv_pool.update_context(context_id, self._active_tokens)
                    self._kv_bridge.export_after_request(
                        context_id, self._active_tokens, self._kv_pool.block_size
                    )
            except Exception:
                self._kv_pool.release_context(context_id)
                self._kv_bridge.release(context_id)
                raise

        response_text = self._postprocess_text(self.tokenizer.decode(generated_ids))
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated_ids),
            "total_tokens": len(prompt_ids) + len(generated_ids),
        }
        if cancel_event.is_set():
            self._session_mgr.clear_stop(context_id)
            return _wrap_completion(context_id, response_text, "stop", usage, stopped=True)
        hit_limit = len(generated_ids) >= max_new_tokens
        finish_reason = "length" if hit_limit else "stop"
        messages = list(messages)
        messages.append({"role": "assistant", "content": response_text})
        self._session_mgr.save_messages(context_id, messages)
        self._session_mgr.clear_stop(context_id)
        return _wrap_completion(context_id, response_text, finish_reason, usage)

    def stream(self, payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        context_id, messages, prompt_ids, sampling, max_new_tokens = self._prepare_request(payload)
        cancel_event = self._session_mgr.get_cancel_event(context_id)
        self._session_mgr.clear_stop(context_id)

        generated_ids: List[int] = []
        filtered = ""
        with self._model_lock:
            acquire = self._kv_pool.acquire_context(context_id, prompt_ids)
            self._kv_bridge.bind_for_request(context_id, prompt_ids, acquire.prefix_len)
            try:
                for token_id in self._iter_generate_ids(
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    sampling=sampling,
                    prefix_len=acquire.prefix_len,
                    cancel_event=cancel_event,
                ):
                    generated_ids.append(int(token_id))
                    new_text = self.tokenizer.decode(generated_ids)
                    new_filtered = self._postprocess_text(new_text)
                    delta = new_filtered[len(filtered) :]
                    filtered = new_filtered
                    if delta:
                        yield _wrap_chunk(context_id, delta, None)
                cancelled = cancel_event.is_set()
                if cancelled:
                    self._active_tokens = list(prompt_ids)
                    self._kv_pool.update_context(context_id, prompt_ids)
                else:
                    self._kv_pool.update_context(context_id, self._active_tokens)
                    self._kv_bridge.export_after_request(
                        context_id, self._active_tokens, self._kv_pool.block_size
                    )
            except Exception:
                self._kv_pool.release_context(context_id)
                self._kv_bridge.release(context_id)
                raise

        if cancel_event.is_set():
            self._session_mgr.clear_stop(context_id)
            usage = {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(prompt_ids) + len(generated_ids),
            }
            yield _wrap_chunk(context_id, None, "stop", usage=usage, stopped=True)
            return

        messages = list(messages)
        messages.append({"role": "assistant", "content": filtered})
        self._session_mgr.save_messages(context_id, messages)
        self._session_mgr.clear_stop(context_id)
        hit_limit = len(generated_ids) >= max_new_tokens
        finish_reason = "length" if hit_limit else "stop"
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated_ids),
            "total_tokens": len(prompt_ids) + len(generated_ids),
        }
        yield _wrap_chunk(context_id, None, finish_reason, usage=usage)

    # ------------------------------------------------------------------
    # Streaming batch API (Phase 1)
    # ------------------------------------------------------------------

    def prepare_batch(self, payloads: List[Dict[str, Any]]) -> Optional[BatchState]:
        """Prefill all sequences in a batch, return BatchState or None to fall back."""
        if not payloads:
            return None
        if not hasattr(self.model, "prefill_packed") or not hasattr(self.model, "step_packed"):
            return None

        sequences: List[BatchSequenceState] = []
        any_sampling = False
        sampling_params_list: List[LlaisysSamplingParams] = []

        for i, payload in enumerate(payloads):
            # Edit-fork requests are not supported in batch path
            if payload.get("edit_from_session_id"):
                return None
            try:
                context_id, messages, prompt_ids, sampling, max_new_tokens = self._prepare_request(payload)
            except Exception:
                return None
            if max_new_tokens <= 0:
                return None

            mode = str(sampling.get("mode", "")).strip().lower()
            top_k = int(sampling.get("top_k", 1))
            top_p = float(sampling.get("top_p", 0.0))
            temperature = float(sampling.get("temperature", 0.0))
            seed = int(sampling.get("seed", 0))
            if mode == "argmax":
                use_sampling = False
            elif mode == "sample":
                use_sampling = True
            else:
                use_sampling = temperature > 0.0 or top_k > 1 or top_p > 0.0
            if use_sampling:
                any_sampling = True

            sp = LlaisysSamplingParams(top_k=top_k, top_p=top_p, temperature=temperature, seed=seed)
            cancel_event = self._session_mgr.get_cancel_event(context_id)
            self._session_mgr.clear_stop(context_id)

            sequences.append(BatchSequenceState(
                index=i,
                context_id=context_id,
                messages=messages,
                prompt_ids=prompt_ids,
                generated_ids=[],
                filtered_text="",
                max_new_tokens=max_new_tokens,
                sampling=sampling,
                sampling_params=sp,
                use_sampling=use_sampling,
                cancel_event=cancel_event,
                finished=False,
                finish_reason=None,
            ))
            sampling_params_list.append(sp)

        # Check for packed-sampling API if needed
        if any_sampling:
            if not hasattr(self.model, "prefill_packed_sampling") or not hasattr(self.model, "step_packed_sampling"):
                return None

        prompts = [seq.prompt_ids for seq in sequences]
        eos = self._eos_token()

        with self._model_lock:
            self.model.reset_kv_cache()
            if any_sampling:
                next_tokens = self.model.prefill_packed_sampling(prompts, sampling_params_list)
            else:
                next_tokens = self.model.prefill_packed(prompts)

        if len(next_tokens) != len(sequences):
            return None

        for i, tok in enumerate(next_tokens):
            t = int(tok)
            if t >= 0:
                sequences[i].generated_ids.append(t)
                # Decode and compute initial filtered text
                new_text = self.tokenizer.decode(sequences[i].generated_ids)
                sequences[i].filtered_text = self._postprocess_text(new_text)
            else:
                sequences[i].finished = True
                sequences[i].finish_reason = "stop"

            # Check immediate termination
            if not sequences[i].finished:
                if eos >= 0 and t == eos:
                    sequences[i].finished = True
                    sequences[i].finish_reason = "stop"
                elif len(sequences[i].generated_ids) >= sequences[i].max_new_tokens:
                    sequences[i].finished = True
                    sequences[i].finish_reason = "length"
                elif sequences[i].cancel_event and sequences[i].cancel_event.is_set():
                    sequences[i].finished = True
                    sequences[i].finish_reason = "stop"

        return BatchState(sequences=sequences, any_sampling=any_sampling, eos_token=eos)

    def step_batch(self, state: BatchState) -> List[StepResult]:
        """Execute one decode step for all active sequences. Dynamic shrinking: skip finished."""
        results: List[StepResult] = []
        active_indices: List[int] = []
        decode_inputs: List[List[int]] = []
        sampling_params_active: List[LlaisysSamplingParams] = []

        for i, seq in enumerate(state.sequences):
            if seq.finished:
                continue
            if seq.cancel_event and seq.cancel_event.is_set():
                seq.finished = True
                seq.finish_reason = "stop"
                results.append(StepResult(
                    seq_index=i, delta_text="", finished=True,
                    finish_reason="stop", stopped=True,
                ))
                continue
            active_indices.append(i)
            last_tok = seq.generated_ids[-1] if seq.generated_ids else 0
            decode_inputs.append([last_tok])
            if seq.sampling_params is not None:
                sampling_params_active.append(seq.sampling_params)

        if not active_indices:
            return results

        with self._model_lock:
            if state.any_sampling:
                step_tokens = self.model.step_packed_sampling(decode_inputs, sampling_params_active)
            else:
                step_tokens = self.model.step_packed(decode_inputs)

        if len(step_tokens) != len(active_indices):
            # Model returned unexpected count; mark all active as finished
            for ai in active_indices:
                seq = state.sequences[ai]
                seq.finished = True
                seq.finish_reason = "stop"
                results.append(StepResult(
                    seq_index=ai, delta_text="", finished=True,
                    finish_reason="stop", stopped=False,
                ))
            return results

        for j, ai in enumerate(active_indices):
            seq = state.sequences[ai]
            t = int(step_tokens[j])

            if t < 0:
                seq.finished = True
                seq.finish_reason = "stop"
                results.append(StepResult(
                    seq_index=ai, delta_text="", finished=True,
                    finish_reason="stop", stopped=False,
                ))
                continue

            seq.generated_ids.append(t)
            new_text = self.tokenizer.decode(seq.generated_ids)
            new_filtered = self._postprocess_text(new_text)
            delta = new_filtered[len(seq.filtered_text):]
            seq.filtered_text = new_filtered

            # Check termination
            finished = False
            finish_reason = None
            stopped = False

            if state.eos_token >= 0 and t == state.eos_token:
                finished = True
                finish_reason = "stop"
            elif len(seq.generated_ids) >= seq.max_new_tokens:
                finished = True
                finish_reason = "length"
            elif seq.cancel_event and seq.cancel_event.is_set():
                finished = True
                finish_reason = "stop"
                stopped = True

            if finished:
                seq.finished = True
                seq.finish_reason = finish_reason

            results.append(StepResult(
                seq_index=ai, delta_text=delta, finished=finished,
                finish_reason=finish_reason, stopped=stopped,
            ))

        return results

    def finalize_sequence(self, state: BatchState, seq_index: int) -> None:
        """Save session history and clean up for a completed sequence."""
        seq = state.sequences[seq_index]
        if seq.cancel_event and seq.cancel_event.is_set():
            self._session_mgr.clear_stop(seq.context_id)
            return
        messages = list(seq.messages)
        messages.append({"role": "assistant", "content": seq.filtered_text})
        self._session_mgr.save_messages(seq.context_id, messages)
        self._session_mgr.clear_stop(seq.context_id)


class ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    scheduler: InferenceScheduler

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._set_cors_headers()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _write_chunk(self, data: bytes) -> bool:
        try:
            self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if parsed.path == "/debug/kv":
            query = parse_qs(parsed.query)
            session_id = str((query.get("session_id") or [""])[0]).strip() or None
            payload = self.scheduler.kv_debug_snapshot(session_id)
            self._send_json(200, payload)
            return
        if parsed.path == "/debug/scheduler":
            self._send_json(200, self.scheduler.debug_snapshot())
            return
        self._send_json(404, _wrap_error("not found", "invalid_request_error", "not_found"))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._set_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path not in ("/chat", "/v1/chat/completions", "/chat/stop"):
            self._send_json(404, _wrap_error("not found", "invalid_request_error", "not_found"))
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json(400, _wrap_error("invalid JSON", "invalid_request_error", "invalid_json"))
            return

        if self.path == "/chat/stop":
            session_id = str(payload.get("session_id") or "").strip()
            if not session_id:
                self._send_json(400, _wrap_error("session_id is required", "invalid_request_error", "missing_field"))
                return
            self.scheduler.request_stop(session_id)
            self._send_json(200, {"ok": True, "session_id": session_id})
            return

        stream = bool(payload.get("stream", False))
        if not stream:
            try:
                handle = self.scheduler.submit(payload, stream=False)
                result = handle.get_result(timeout=self.scheduler.request_timeout_seconds())
                if isinstance(result, dict) and result.get("error"):
                    code = 504 if result.get("code") == "timeout" else 400
                    err = result.get("error")
                    err_code = str(result.get("code", "")) or "server_error"
                    self._send_json(code, _wrap_error(str(err), "server_error", err_code))
                    return
            except SchedulerQueueFullError as exc:
                self._send_json(429, _wrap_error(str(exc), "server_error", "queue_full"))
                return
            except TaskTimeoutError as exc:
                self._send_json(504, _wrap_error(str(exc), "server_error", "timeout"))
                return
            except RuntimeError as exc:
                self._send_json(400, _wrap_error(str(exc), "server_error"))
                return
            self._send_json(200, result)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self._set_cors_headers()
        self.end_headers()

        current_session_id = ""
        try:
            handle = self.scheduler.submit(payload, stream=True)
            for item in handle.iter_stream(timeout=self.scheduler.request_timeout_seconds()):
                current_session_id = str(item.get("session_id") or current_session_id)
                data = json.dumps(item, ensure_ascii=False).encode("utf-8")
                if not self._write_chunk(b"data: " + data + b"\n\n"):
                    if current_session_id:
                        self.scheduler.request_stop(current_session_id)
                    return
            self._write_chunk(b"data: [DONE]\n\n")
        except SchedulerQueueFullError as exc:
            err = _wrap_error(str(exc), "server_error", "queue_full")
            data = json.dumps(err, ensure_ascii=False).encode("utf-8")
            self._write_chunk(b"data: " + data + b"\n\n")
        except TaskTimeoutError as exc:
            if current_session_id:
                self.scheduler.request_stop(current_session_id)
            err = _wrap_error(str(exc), "server_error", "timeout")
            data = json.dumps(err, ensure_ascii=False).encode("utf-8")
            self._write_chunk(b"data: " + data + b"\n\n")
        except Exception as exc:
            if current_session_id:
                self.scheduler.request_stop(current_session_id)
            err = _wrap_error(str(exc), "server_error")
            data = json.dumps(err, ensure_ascii=False).encode("utf-8")
            self._write_chunk(b"data: " + data + b"\n\n")
        finally:
            self._write_chunk(b"")


def _resolve_tokenizer_path(model_path: str, tokenizer_path: Optional[str]) -> str:
    if tokenizer_path:
        return tokenizer_path
    path = Path(model_path)
    sp = path / "tokenizer.model"
    if sp.exists():
        return str(sp)
    hf = path / "tokenizer.json"
    if hf.exists():
        return str(hf)
    raise FileNotFoundError(f"No tokenizer.model or tokenizer.json found under: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="model directory")
    parser.add_argument("--tokenizer", required=False, type=str, help="tokenizer file path")
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--pool-size", default=1, type=int, help="deprecated")
    parser.add_argument(
        "--kv-runtime-reuse",
        action="store_true",
        help="enable experimental runtime KV reuse fast-path",
    )
    parser.add_argument("--kv-block-size", default=64, type=int, help="kv block token size")
    parser.add_argument("--kv-max-blocks", default=4096, type=int, help="kv max block count")
    parser.add_argument("--kv-max-bytes", default=268435456, type=int, help="kv max bytes")
    parser.add_argument("--workers", default=1, type=int, help="inference worker count")
    parser.add_argument("--queue-size", default=128, type=int, help="max queued tasks per worker")
    parser.add_argument("--request-timeout-ms", default=120000, type=int, help="scheduler request timeout in milliseconds")
    parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="enable minimal iteration-level continuous scheduling",
    )
    parser.add_argument(
        "--kv-aware-routing",
        action="store_true",
        help="enable KV-aware worker routing (query KV pool before dispatching)",
    )
    parser.add_argument(
        "--max-batch-size",
        default=8,
        type=int,
        help="max sequences per streaming batch (default 8)",
    )
    parser.add_argument(
        "--shared-model",
        action="store_true",
        help="share a single model instance and KV pool across all workers",
    )
    parser.add_argument(
        "--kv-memory-threshold",
        default=0.0,
        type=float,
        help="KV memory pressure threshold (0.0=disabled, 0.85=recommended)",
    )
    args = parser.parse_args()

    tokenizer_path = _resolve_tokenizer_path(args.model, args.tokenizer)
    worker_count = max(1, int(args.workers))
    services: List[ChatService] = []
    if args.shared_model:
        # Shared mode: one model, one tokenizer, one KV pool, one KV bridge, one lock
        model = Qwen2(
            args.model,
            llaisys.DeviceType.CPU if args.device == "cpu" else llaisys.DeviceType.NVIDIA,
        )
        tokenizer = llaisys.Tokenizer(tokenizer_path)
        shared_lock = threading.RLock()
        shared_kv_pool = KVCachePool(
            block_size=args.kv_block_size,
            max_blocks=args.kv_max_blocks,
            max_bytes=args.kv_max_bytes,
        )
        shared_kv_bridge = KVRuntimeBridge(model, enabled=args.kv_runtime_reuse)
        for _ in range(worker_count):
            services.append(
                ChatService(
                    model,
                    tokenizer,
                    model_path=args.model,
                    enable_kv_runtime_reuse=args.kv_runtime_reuse,
                    block_size=args.kv_block_size,
                    max_blocks=args.kv_max_blocks,
                    max_bytes=args.kv_max_bytes,
                    model_lock=shared_lock,
                    kv_pool=shared_kv_pool,
                    kv_bridge=shared_kv_bridge,
                )
            )
    else:
        for _ in range(worker_count):
            tokenizer = llaisys.Tokenizer(tokenizer_path)
            model = Qwen2(
                args.model,
                llaisys.DeviceType.CPU if args.device == "cpu" else llaisys.DeviceType.NVIDIA,
            )
            services.append(
                ChatService(
                    model,
                    tokenizer,
                    model_path=args.model,
                    enable_kv_runtime_reuse=args.kv_runtime_reuse,
                    block_size=args.kv_block_size,
                    max_blocks=args.kv_max_blocks,
                    max_bytes=args.kv_max_bytes,
                )
            )
    scheduler = InferenceScheduler(
        services,
        queue_size=max(1, int(args.queue_size)),
        request_timeout_ms=max(0, int(args.request_timeout_ms)),
        continuous_batching=bool(args.continuous_batching),
        kv_aware_routing=bool(args.kv_aware_routing),
        max_batch_size=max(1, int(args.max_batch_size)),
        kv_memory_threshold=float(args.kv_memory_threshold),
    )
    scheduler.start()

    handler = ChatHandler
    handler.scheduler = scheduler
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    kv_routing_str = ", kv_aware_routing=on" if args.kv_aware_routing else ""
    shared_str = ", shared_model=on" if args.shared_model else ""
    kv_mem_str = f", kv_memory_threshold={args.kv_memory_threshold}" if args.kv_memory_threshold > 0 else ""
    print(
        f"LLAISYS chat server listening on http://{args.host}:{args.port} "
        f"(workers={worker_count}, queue_size={max(1, int(args.queue_size))}{kv_routing_str}{shared_str}{kv_mem_str})"
    )
    try:
        server.serve_forever()
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
