from __future__ import annotations

from dataclasses import dataclass
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import OrderedDict, deque

if TYPE_CHECKING:
    from llaisys.interfaces import IInferenceService

logger = logging.getLogger(__name__)

_END = object()


@dataclass
class InferenceTask:
    payload: Dict[str, Any]
    stream: bool
    output_queue: "queue.Queue[Any]"
    deadline_at: Optional[float]


@dataclass
class _ActiveTask:
    task: InferenceTask
    iterator: Any
    emitted_any: bool = False


class SchedulerQueueFullError(RuntimeError):
    pass


class TaskTimeoutError(RuntimeError):
    pass


class TaskHandle:
    def __init__(self, output_queue: "queue.Queue[Any]") -> None:
        self._q = output_queue

    def get_result(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        while True:
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty as exc:
                raise TaskTimeoutError("task result timeout") from exc
            if item is _END:
                raise RuntimeError("task ended without result")
            if isinstance(item, dict):
                return item
            raise RuntimeError("unexpected task result type")

    def iter_stream(self, timeout: Optional[float] = None) -> Iterable[Dict[str, Any]]:
        while True:
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty as exc:
                raise TaskTimeoutError("task stream timeout") from exc
            if item is _END:
                break
            if isinstance(item, dict):
                yield item
            else:
                raise RuntimeError("unexpected stream item type")


class InferenceScheduler:
    """In-process scheduler with per-worker queues and session stickiness."""

    def __init__(
        self,
        services: "List[IInferenceService]",
        queue_size: int = 128,
        request_timeout_ms: int = 120000,
        continuous_batching: bool = False,
        kv_aware_routing: bool = False,
        max_sticky_sessions: int = 10000,
        max_batch_size: int = 8,
        kv_memory_threshold: float = 0.0,
    ) -> None:
        if not services:
            raise ValueError("services must not be empty")
        self._services: "List[IInferenceService]" = list(services)
        self._queue_size = max(1, int(queue_size))
        self._request_timeout_ms = max(0, int(request_timeout_ms))
        self._continuous_batching = bool(continuous_batching)
        self._kv_aware_routing = bool(kv_aware_routing)
        self._max_sticky_sessions = max(100, int(max_sticky_sessions))
        self._max_batch_size = max(1, int(max_batch_size))
        self._kv_memory_threshold = float(kv_memory_threshold)
        self._queues: List["queue.Queue[Optional[InferenceTask]]"] = [
            queue.Queue(maxsize=self._queue_size) for _ in self._services
        ]
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._session_worker: "OrderedDict[str, int]" = OrderedDict()
        self._rr = 0
        self._packed_prefill_last_error: str = ""
        self._metrics: Dict[str, float] = {
            "submitted": 0.0,
            "completed": 0.0,
            "cancelled": 0.0,
            "failed": 0.0,
            "timed_out": 0.0,
            "queue_full": 0.0,
            "stop_requests": 0.0,
            "batch_rounds": 0.0,
            "batch_active_sum": 0.0,
            "batch_last_active": 0.0,
            "prefill_rounds": 0.0,
            "decode_rounds": 0.0,
            "prefill_last_active": 0.0,
            "decode_last_active": 0.0,
            "packed_prefill_batches": 0.0,
            "packed_prefill_tasks": 0.0,
            "packed_prefill_attempts": 0.0,
            "packed_prefill_candidate_tasks": 0.0,
            "packed_prefill_none_returns": 0.0,
            "packed_prefill_exceptions": 0.0,
            # KV 感知路由指标
            "kv_aware_routing_attempts": 0.0,
            "kv_aware_routing_hits": 0.0,
            "kv_aware_routing_best_prefix_len_sum": 0.0,
            # 流式批处理指标
            "stream_batch_prefill_batches": 0.0,
            "stream_batch_prefill_tasks": 0.0,
            "stream_batch_decode_rounds": 0.0,
            "stream_batch_decode_active_sum": 0.0,
            "stream_batch_shrink_events": 0.0,
            "stream_batch_fallback_tasks": 0.0,
            # KV 内存流控指标
            "kv_memory_rejected": 0.0,
        }

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for idx in range(len(self._services)):
            t = threading.Thread(target=self._worker_loop, args=(idx,), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for q in self._queues:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()

    def submit(self, payload: Dict[str, Any], stream: bool) -> TaskHandle:
        payload = dict(payload)  # shallow copy to avoid mutating caller's dict

        # KV 内存感知流控：超过阈值时拒绝新请求
        if self._kv_memory_threshold > 0:
            try:
                pressure = max(
                    svc.kv_pool.memory_pressure() for svc in self._services
                )
            except Exception:
                pressure = 0.0
            if pressure > self._kv_memory_threshold:
                with self._lock:
                    self._metrics["kv_memory_rejected"] += 1.0
                raise SchedulerQueueFullError("KV memory pressure too high")

        # 自动 tokenize：如果启用了 KV 感知路由且 payload 中没有 _prompt_tokens
        if (
            self._kv_aware_routing
            and "_prompt_tokens" not in payload
            and len(self._services) > 1
        ):
            try:
                # 使用第一个服务进行 tokenize（所有服务使用相同的 tokenizer）
                svc = self._services[0]
                if hasattr(svc, "tokenize_for_routing"):
                    tokens = svc.tokenize_for_routing(payload)
                    if tokens:
                        payload["_prompt_tokens"] = tokens
            except Exception:
                logger.debug("tokenize_for_routing failed, falling back to default routing", exc_info=True)

        worker_idx = self._choose_worker(payload)

        # 清理路由专用的内部字段，不传递给下游
        payload.pop("_prompt_tokens", None)

        out_q: "queue.Queue[Any]" = queue.Queue()
        deadline_at: Optional[float] = None
        if self._request_timeout_ms > 0:
            deadline_at = time.time() + self._request_timeout_ms / 1000.0
        task = InferenceTask(payload=payload, stream=bool(stream), output_queue=out_q, deadline_at=deadline_at)
        try:
            self._queues[worker_idx].put_nowait(task)
        except queue.Full:
            with self._lock:
                self._metrics["queue_full"] += 1.0
            raise SchedulerQueueFullError("scheduler queue is full")
        with self._lock:
            self._metrics["submitted"] += 1.0
        return TaskHandle(out_q)

    def request_stop(self, session_id: str) -> bool:
        sid = str(session_id or "").strip()
        if not sid:
            return False
        with self._lock:
            self._metrics["stop_requests"] += 1.0
            idx = self._session_worker.get(sid)
        if idx is not None:
            return bool(self._services[idx].request_stop(sid))
        ok = False
        for svc in self._services:
            ok = bool(svc.request_stop(sid)) or ok
        return ok

    def kv_debug_snapshot(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = str(session_id or "").strip()
        if sid:
            with self._lock:
                idx = self._session_worker.get(sid)
            if idx is not None:
                snap = self._services[idx].kv_debug_snapshot(sid)
                snap["worker"] = idx
                return snap
            for idx2, svc in enumerate(self._services):
                snap = svc.kv_debug_snapshot(sid)
                if snap.get("has_native_context") or snap.get("last_bind"):
                    snap["worker"] = idx2
                    return snap
            return self._services[0].kv_debug_snapshot(sid)

        merged = {
            "session_id": None,
            "workers": len(self._services),
            "queue_size": self._queue_size,
            "queues": [q.qsize() for q in self._queues],
            "kv_pool": {
                "contexts": 0.0,
                "blocks": 0.0,
                "prefix_entries": 0.0,
                "total_bytes": 0.0,
                "zero_ref_blocks": 0.0,
                "shared_blocks": 0.0,
                "total_refs": 0.0,
                "acquire_count": 0.0,
                "prefix_hit_count": 0.0,
                "prefix_hit_rate": 0.0,
                "avg_matched_tokens": 0.0,
            },
        }

        # 共享 KV 池优化：所有 worker 共享同一个池时只查询一次
        first_pool = getattr(self._services[0], "kv_pool", None)
        shared_pool = first_pool is not None and all(
            getattr(svc, "kv_pool", None) is first_pool for svc in self._services[1:]
        )

        if shared_pool:
            snap = self._services[0].kv_debug_snapshot(None)
            pool = snap.get("kv_pool", {})
            for k in merged["kv_pool"]:
                merged["kv_pool"][k] = float(pool.get(k, 0.0))
        else:
            hit_rate_numer = 0.0
            hit_rate_denom = 0.0
            matched_numer = 0.0
            matched_denom = 0.0
            for svc in self._services:
                snap = svc.kv_debug_snapshot(None)
                pool = snap.get("kv_pool", {})
                for k in ("contexts", "blocks", "prefix_entries", "total_bytes", "zero_ref_blocks", "shared_blocks", "total_refs", "acquire_count", "prefix_hit_count"):
                    merged["kv_pool"][k] += float(pool.get(k, 0.0))
                hit_rate_numer += float(pool.get("prefix_hit_count", 0.0))
                hit_rate_denom += float(pool.get("acquire_count", 0.0))
                matched_numer += float(pool.get("avg_matched_tokens", 0.0)) * float(pool.get("acquire_count", 0.0))
                matched_denom += float(pool.get("acquire_count", 0.0))
            merged["kv_pool"]["prefix_hit_rate"] = hit_rate_numer / hit_rate_denom if hit_rate_denom > 0 else 0.0
            merged["kv_pool"]["avg_matched_tokens"] = matched_numer / matched_denom if matched_denom > 0 else 0.0
        return merged

    def debug_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            metrics = dict(self._metrics)
            packed_prefill_last_error = self._packed_prefill_last_error
            sticky_sessions = len(self._session_worker)
        avg_batch_active = (
            metrics.get("batch_active_sum", 0.0) / metrics.get("batch_rounds", 1.0)
            if metrics.get("batch_rounds", 0.0) > 0
            else 0.0
        )
        kv_routing_attempts = metrics.get("kv_aware_routing_attempts", 0.0)
        kv_routing_hit_rate = (
            metrics.get("kv_aware_routing_hits", 0.0) / kv_routing_attempts
            if kv_routing_attempts > 0
            else 0.0
        )
        kv_routing_avg_prefix_len = (
            metrics.get("kv_aware_routing_best_prefix_len_sum", 0.0) / metrics.get("kv_aware_routing_hits", 1.0)
            if metrics.get("kv_aware_routing_hits", 0.0) > 0
            else 0.0
        )
        # KV memory pressure snapshot
        try:
            kv_memory_pressure = max(
                svc.kv_pool.memory_pressure() for svc in self._services
            )
        except Exception:
            kv_memory_pressure = 0.0
        return {
            "workers": len(self._services),
            "queue_size": self._queue_size,
            "queues": [q.qsize() for q in self._queues],
            "request_timeout_ms": self._request_timeout_ms,
            "continuous_batching": self._continuous_batching,
            "kv_aware_routing": self._kv_aware_routing,
            "kv_routing_hit_rate": kv_routing_hit_rate,
            "kv_routing_avg_prefix_len": kv_routing_avg_prefix_len,
            "kv_memory_threshold": self._kv_memory_threshold,
            "kv_memory_pressure": kv_memory_pressure,
            "max_batch_size": self._max_batch_size,
            "avg_batch_active": avg_batch_active,
            "sticky_sessions": sticky_sessions,
            "packed_prefill_last_error": packed_prefill_last_error,
            "metrics": metrics,
        }

    def request_timeout_seconds(self) -> Optional[float]:
        if self._request_timeout_ms <= 0:
            return None
        return self._request_timeout_ms / 1000.0

    def _touch_session(self, sid: str, worker_idx: int) -> None:
        """Record/update session->worker mapping with LRU eviction. Caller must hold self._lock."""
        if sid in self._session_worker:
            self._session_worker.move_to_end(sid)
        self._session_worker[sid] = worker_idx
        while len(self._session_worker) > self._max_sticky_sessions:
            self._session_worker.popitem(last=False)

    def _choose_worker(self, payload: Dict[str, Any]) -> int:
        sid = str(payload.get("session_id") or payload.get("edit_from_session_id") or "").strip()

        # 1. 会话粘性：已绑定的 session 优先路由到原 worker
        with self._lock:
            if sid and sid in self._session_worker:
                self._session_worker.move_to_end(sid)
                return self._session_worker[sid]

        # 2. KV 感知路由：查询各 worker 的 KV 命中情况
        # KV 感知路由是 best-effort：查询到入队之间 KV 状态可能变化，
        # 最坏情况是路由到非最优 worker，不影响正确性。
        prompt_tokens: Optional[Sequence[int]] = payload.get("_prompt_tokens")
        if self._kv_aware_routing and prompt_tokens and len(self._services) > 1:
            best_worker = -1
            best_prefix_len = -1

            # 共享 KV 池优化：所有 worker 共享同一个池时只查询一次
            first_pool = getattr(self._services[0], "kv_pool", None)
            shared_pool = first_pool is not None and all(
                getattr(svc, "kv_pool", None) is first_pool for svc in self._services[1:]
            )

            if shared_pool:
                try:
                    prefix_len = first_pool.query_prefix_len(prompt_tokens)
                    if prefix_len > 0:
                        best_prefix_len = prefix_len
                        # 共享池模式下选负载最轻的 worker
                        best_worker = min(range(len(self._queues)), key=lambda i: self._queues[i].qsize())
                except Exception:
                    pass
            else:
                for idx, svc in enumerate(self._services):
                    try:
                        kv_pool = getattr(svc, "kv_pool", None)
                        if kv_pool is None:
                            continue
                        prefix_len = kv_pool.query_prefix_len(prompt_tokens)
                        if prefix_len > best_prefix_len:
                            best_prefix_len = prefix_len
                            best_worker = idx
                    except Exception:
                        # 查询失败，跳过该 worker
                        continue

            with self._lock:
                self._metrics["kv_aware_routing_attempts"] += 1.0
                if best_prefix_len > 0:
                    self._metrics["kv_aware_routing_hits"] += 1.0
                    self._metrics["kv_aware_routing_best_prefix_len_sum"] += float(best_prefix_len)

            if best_worker >= 0 and best_prefix_len > 0:
                if sid:
                    with self._lock:
                        self._touch_session(sid, best_worker)
                return best_worker

        # 3. Fallback：hash 或轮询
        with self._lock:
            if sid:
                idx = hash(sid) % len(self._services)
                self._touch_session(sid, idx)
                return idx
            idx = self._rr % len(self._services)
            self._rr = (self._rr + 1) % len(self._services)
            return idx

    def _bind_session(self, session_id: Optional[str], worker_idx: int) -> None:
        sid = str(session_id or "").strip()
        if not sid:
            return
        with self._lock:
            self._touch_session(sid, worker_idx)

    def _worker_loop(self, idx: int) -> None:
        if self._continuous_batching:
            self._worker_loop_continuous(idx)
            return

        svc = self._services[idx]
        q = self._queues[idx]
        while not self._stop.is_set():
            task = q.get()
            if task is None:
                q.task_done()
                continue
            try:
                if task.deadline_at is not None and time.time() > task.deadline_at:
                    with self._lock:
                        self._metrics["timed_out"] += 1.0
                    if task.stream:
                        task.output_queue.put({"error": "request timeout", "code": "timeout", "done": True})
                    else:
                        task.output_queue.put({"error": "request timeout", "code": "timeout"})
                    task.output_queue.put(_END)
                    continue
                if task.stream:
                    try:
                        for item in svc.stream(task.payload):
                            if isinstance(item, dict):
                                self._bind_session(item.get("session_id"), idx)
                                if item.get("done") and item.get("stopped"):
                                    with self._lock:
                                        self._metrics["cancelled"] += 1.0
                            task.output_queue.put(item)
                        with self._lock:
                            self._metrics["completed"] += 1.0
                    except Exception as exc:
                        with self._lock:
                            self._metrics["failed"] += 1.0
                        task.output_queue.put({"error": str(exc), "done": True})
                    finally:
                        task.output_queue.put(_END)
                else:
                    try:
                        result = svc.generate(task.payload)
                        if isinstance(result, dict):
                            self._bind_session(result.get("session_id"), idx)
                        task.output_queue.put(result)
                        with self._lock:
                            self._metrics["completed"] += 1.0
                            if isinstance(result, dict) and result.get("stopped"):
                                self._metrics["cancelled"] += 1.0
                    except Exception as exc:
                        with self._lock:
                            self._metrics["failed"] += 1.0
                        task.output_queue.put({"error": str(exc)})
                    finally:
                        task.output_queue.put(_END)
            finally:
                q.task_done()

    def _worker_loop_continuous(self, idx: int) -> None:
        svc = self._services[idx]
        q = self._queues[idx]
        # Raw tasks waiting for prefill (not yet started)
        prefill_pending: "deque[InferenceTask]" = deque()
        # Fallback: legacy iterator-based active tasks
        fallback_prefill: "deque[_ActiveTask]" = deque()
        fallback_decode: List[_ActiveTask] = []
        # Batch-driven decode state (from prepare_batch)
        batch_state: Optional[Any] = None
        batch_tasks: List[InferenceTask] = []  # parallel to batch_state.sequences

        def _append_from_queue(block: bool) -> None:
            while True:
                try:
                    task = q.get(block=block, timeout=0.1 if block else 0.0)
                except queue.Empty:
                    return
                if task is None:
                    q.task_done()
                    return
                prefill_pending.append(task)
                q.task_done()
                block = False

        def _emit_chunk(task: InferenceTask, chunk: dict) -> None:
            """Send a stream chunk or accumulate for non-stream."""
            task.output_queue.put(chunk)

        def _emit_final_stream(task: InferenceTask, context_id: str,
                               finish_reason: str, prompt_len: int,
                               gen_len: int, stopped: bool) -> None:
            usage = {
                "prompt_tokens": prompt_len,
                "completion_tokens": gen_len,
                "total_tokens": prompt_len + gen_len,
            }
            from llaisys.server import _wrap_chunk
            chunk = _wrap_chunk(context_id, None, finish_reason, usage=usage, stopped=stopped)
            task.output_queue.put(chunk)
            task.output_queue.put(_END)

        def _emit_final_non_stream(task: InferenceTask, context_id: str,
                                   content: str, finish_reason: str,
                                   prompt_len: int, gen_len: int,
                                   stopped: bool) -> None:
            usage = {
                "prompt_tokens": prompt_len,
                "completion_tokens": gen_len,
                "total_tokens": prompt_len + gen_len,
            }
            from llaisys.server import _wrap_completion
            result = _wrap_completion(context_id, content, finish_reason, usage, stopped=stopped)
            task.output_queue.put(result)
            task.output_queue.put(_END)

        # --- Fallback helpers (legacy iterator path) ---
        def _step_once(state: _ActiveTask) -> str:
            task = state.task
            it = state.iterator
            if task.deadline_at is not None and time.time() > task.deadline_at:
                with self._lock:
                    self._metrics["timed_out"] += 1.0
                if task.stream:
                    task.output_queue.put({"error": "request timeout", "code": "timeout", "done": True})
                else:
                    task.output_queue.put({"error": "request timeout", "code": "timeout"})
                task.output_queue.put(_END)
                return "done"
            try:
                item = next(it)
                if isinstance(item, dict):
                    self._bind_session(item.get("session_id"), idx)

                def _is_final(d: dict) -> bool:
                    if d.get("done"):
                        return True
                    choices = d.get("choices")
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        if choices[0].get("finish_reason") is not None:
                            return True
                    return False

                def _is_stopped(d: dict) -> bool:
                    if d.get("stopped"):
                        return True
                    choices = d.get("choices")
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        if choices[0].get("finish_reason") == "stop":
                            return True
                    return False

                if task.stream:
                    if not isinstance(item, dict):
                        raise RuntimeError("stream item must be dict")
                    task.output_queue.put(item)
                    state.emitted_any = True
                    if _is_final(item):
                        with self._lock:
                            self._metrics["completed"] += 1.0
                            if _is_stopped(item):
                                self._metrics["cancelled"] += 1.0
                        task.output_queue.put(_END)
                        return "done"
                    return "keep"
                if isinstance(item, dict) and _is_final(item):
                    if item.get("error"):
                        with self._lock:
                            self._metrics["failed"] += 1.0
                        task.output_queue.put({"error": str(item.get("error"))})
                    else:
                        result = dict(item)
                        choices = result.get("choices")
                        if choices and isinstance(choices, list) and len(choices) > 0:
                            c = dict(choices[0])
                            acc = getattr(state, "accumulated_content", "")
                            delta = c.pop("delta", {})
                            final_content = acc + delta.get("content", "")
                            c["message"] = {"role": "assistant", "content": final_content}
                            result["choices"] = [c]
                        if result.get("object") == "chat.completion.chunk":
                            result["object"] = "chat.completion"
                        task.output_queue.put(result)
                        with self._lock:
                            self._metrics["completed"] += 1.0
                            if _is_stopped(item):
                                self._metrics["cancelled"] += 1.0
                    task.output_queue.put(_END)
                    return "done"
                choices = item.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        state.accumulated_content = getattr(state, "accumulated_content", "") + content
                return "keep"
            except StopIteration:
                with self._lock:
                    self._metrics["failed"] += 1.0
                if task.stream:
                    task.output_queue.put({"error": "stream ended unexpectedly", "done": True})
                else:
                    task.output_queue.put({"error": "task ended unexpectedly"})
                task.output_queue.put(_END)
                return "done"
            except Exception as exc:
                with self._lock:
                    self._metrics["failed"] += 1.0
                if task.stream:
                    task.output_queue.put({"error": str(exc), "done": True})
                else:
                    task.output_queue.put({"error": str(exc)})
                task.output_queue.put(_END)
                return "done"

        while not self._stop.is_set():
            has_work = (
                prefill_pending or fallback_prefill or fallback_decode
                or (batch_state is not None)
            )
            if not has_work:
                _append_from_queue(block=True)
                if not prefill_pending:
                    continue
            else:
                _append_from_queue(block=False)

            with self._lock:
                total_active = (
                    len(prefill_pending) + len(fallback_prefill) + len(fallback_decode)
                    + (len([s for s in batch_state.sequences if not s.finished]) if batch_state else 0)
                )
                self._metrics["batch_rounds"] += 1.0
                self._metrics["batch_active_sum"] += float(total_active)
                self._metrics["batch_last_active"] = float(total_active)
                self._metrics["prefill_last_active"] = float(len(prefill_pending) + len(fallback_prefill))
                decode_count = len(fallback_decode) + (
                    len([s for s in batch_state.sequences if not s.finished]) if batch_state else 0
                )
                self._metrics["decode_last_active"] = float(decode_count)

            # ============================================================
            # P stage: try batch prefill for pending tasks
            # ============================================================
            if prefill_pending and batch_state is None:
                with self._lock:
                    self._metrics["prefill_rounds"] += 1.0

                # Collect candidates up to max_batch_size
                batch_candidates: List[InferenceTask] = []
                remaining: "deque[InferenceTask]" = deque()
                decode_active_count = len(fallback_decode)
                slots = self._max_batch_size - decode_active_count

                while prefill_pending and len(batch_candidates) < slots:
                    task = prefill_pending.popleft()
                    # Check deadline
                    if task.deadline_at is not None and time.time() > task.deadline_at:
                        with self._lock:
                            self._metrics["timed_out"] += 1.0
                        if task.stream:
                            task.output_queue.put({"error": "request timeout", "code": "timeout", "done": True})
                        else:
                            task.output_queue.put({"error": "request timeout", "code": "timeout"})
                        task.output_queue.put(_END)
                        continue
                    batch_candidates.append(task)

                if not batch_candidates:
                    pass  # all timed out
                elif len(batch_candidates) >= 1 and hasattr(svc, "prepare_batch"):
                    # Try batch path
                    try:
                        payloads = [t.payload for t in batch_candidates]
                        result = svc.prepare_batch(payloads)
                    except Exception as exc:
                        logger.debug("prepare_batch failed: %s", exc, exc_info=True)
                        result = None

                    if result is not None:
                        batch_state = result
                        batch_tasks = list(batch_candidates)
                        with self._lock:
                            self._metrics["stream_batch_prefill_batches"] += 1.0
                            self._metrics["stream_batch_prefill_tasks"] += float(len(batch_candidates))

                        # Emit first token chunks for each sequence
                        from llaisys.server import _wrap_chunk
                        for i, seq in enumerate(batch_state.sequences):
                            task = batch_tasks[i]
                            self._bind_session(seq.context_id, idx)
                            if seq.filtered_text and task.stream:
                                chunk = _wrap_chunk(seq.context_id, seq.filtered_text, None)
                                task.output_queue.put(chunk)
                            # If already finished after prefill
                            if seq.finished:
                                if task.stream:
                                    _emit_final_stream(
                                        task, seq.context_id, seq.finish_reason or "stop",
                                        len(seq.prompt_ids), len(seq.generated_ids),
                                        stopped=bool(seq.cancel_event and seq.cancel_event.is_set()),
                                    )
                                else:
                                    _emit_final_non_stream(
                                        task, seq.context_id, seq.filtered_text,
                                        seq.finish_reason or "stop",
                                        len(seq.prompt_ids), len(seq.generated_ids),
                                        stopped=bool(seq.cancel_event and seq.cancel_event.is_set()),
                                    )
                                with self._lock:
                                    self._metrics["completed"] += 1.0
                                svc.finalize_sequence(batch_state, i)
                        # If all finished after prefill, clear batch
                        if all(s.finished for s in batch_state.sequences):
                            batch_state = None
                            batch_tasks = []
                    else:
                        # Fallback: push to legacy iterator path
                        with self._lock:
                            self._metrics["stream_batch_fallback_tasks"] += float(len(batch_candidates))
                        for task in batch_candidates:
                            try:
                                it = svc.stream(task.payload)
                                fallback_prefill.append(_ActiveTask(task=task, iterator=it))
                            except Exception as exc:
                                if task.stream:
                                    task.output_queue.put({"error": str(exc), "done": True})
                                else:
                                    task.output_queue.put({"error": str(exc)})
                                task.output_queue.put(_END)
                                with self._lock:
                                    self._metrics["failed"] += 1.0
                else:
                    # No prepare_batch available, use legacy path
                    with self._lock:
                        self._metrics["stream_batch_fallback_tasks"] += float(len(batch_candidates))
                    for task in batch_candidates:
                        try:
                            it = svc.stream(task.payload)
                            fallback_prefill.append(_ActiveTask(task=task, iterator=it))
                        except Exception as exc:
                            if task.stream:
                                task.output_queue.put({"error": str(exc), "done": True})
                            else:
                                task.output_queue.put({"error": str(exc)})
                            task.output_queue.put(_END)
                            with self._lock:
                                self._metrics["failed"] += 1.0

            # Legacy P stage: step fallback prefill tasks one at a time
            if fallback_prefill:
                with self._lock:
                    if not prefill_pending:
                        self._metrics["prefill_rounds"] += 1.0

                # Try packed prefill for non-stream fallback tasks
                packed_candidates: List[_ActiveTask] = []
                for state in fallback_prefill:
                    if state.task.stream:
                        continue
                    packed_candidates.append(state)
                    if len(packed_candidates) >= self._max_batch_size:
                        break
                if len(packed_candidates) >= 2 and (
                    hasattr(svc, "generate_packed_non_stream") or hasattr(svc, "generate_packed_once")
                ):
                    packed_exception = False
                    with self._lock:
                        self._metrics["packed_prefill_attempts"] += 1.0
                        self._metrics["packed_prefill_candidate_tasks"] += float(len(packed_candidates))
                    try:
                        packed_payloads = [st.task.payload for st in packed_candidates]
                        if hasattr(svc, "generate_packed_non_stream"):
                            packed_results = svc.generate_packed_non_stream(packed_payloads)
                        else:
                            packed_results = svc.generate_packed_once(packed_payloads)
                    except Exception as exc:
                        packed_exception = True
                        with self._lock:
                            self._metrics["packed_prefill_exceptions"] += 1.0
                            self._packed_prefill_last_error = str(exc)
                        packed_results = None
                    if isinstance(packed_results, list) and len(packed_results) == len(packed_candidates):
                        packed_ids = {id(st) for st in packed_candidates}
                        fallback_prefill = deque([st for st in fallback_prefill if id(st) not in packed_ids])
                        for st, result in zip(packed_candidates, packed_results):
                            st.task.output_queue.put(result)
                            st.task.output_queue.put(_END)
                        with self._lock:
                            self._metrics["completed"] += float(len(packed_candidates))
                            self._metrics["packed_prefill_batches"] += 1.0
                            self._metrics["packed_prefill_tasks"] += float(len(packed_candidates))
                            self._packed_prefill_last_error = ""
                        # Skip single step below if we consumed all
                        if not fallback_prefill:
                            pass
                    elif not packed_exception:
                        with self._lock:
                            self._metrics["packed_prefill_none_returns"] += 1.0

                if fallback_prefill:
                    state = fallback_prefill.popleft()
                    status = _step_once(state)
                    if status == "keep":
                        fallback_decode.append(state)

            # ============================================================
            # D stage: batch decode
            # ============================================================
            if batch_state is not None:
                active_before = len([s for s in batch_state.sequences if not s.finished])
                with self._lock:
                    self._metrics["decode_rounds"] += 1.0
                    self._metrics["stream_batch_decode_rounds"] += 1.0
                    self._metrics["stream_batch_decode_active_sum"] += float(active_before)

                try:
                    step_results = svc.step_batch(batch_state)
                except Exception as exc:
                    logger.debug("step_batch failed: %s", exc, exc_info=True)
                    # Mark all active as failed
                    for i, seq in enumerate(batch_state.sequences):
                        if not seq.finished:
                            seq.finished = True
                            task = batch_tasks[i]
                            if task.stream:
                                task.output_queue.put({"error": str(exc), "done": True})
                            else:
                                task.output_queue.put({"error": str(exc)})
                            task.output_queue.put(_END)
                            with self._lock:
                                self._metrics["failed"] += 1.0
                    batch_state = None
                    batch_tasks = []
                    step_results = None

                if step_results is not None:
                    from llaisys.server import _wrap_chunk
                    for sr in step_results:
                        task = batch_tasks[sr.seq_index]
                        seq = batch_state.sequences[sr.seq_index]

                        if sr.delta_text and task.stream:
                            chunk = _wrap_chunk(seq.context_id, sr.delta_text, None)
                            task.output_queue.put(chunk)

                        if sr.finished:
                            if task.stream:
                                _emit_final_stream(
                                    task, seq.context_id, sr.finish_reason or "stop",
                                    len(seq.prompt_ids), len(seq.generated_ids),
                                    stopped=sr.stopped,
                                )
                            else:
                                _emit_final_non_stream(
                                    task, seq.context_id, seq.filtered_text,
                                    sr.finish_reason or "stop",
                                    len(seq.prompt_ids), len(seq.generated_ids),
                                    stopped=sr.stopped,
                                )
                            with self._lock:
                                self._metrics["completed"] += 1.0
                                if sr.stopped:
                                    self._metrics["cancelled"] += 1.0
                            svc.finalize_sequence(batch_state, sr.seq_index)

                    # Check for shrink events
                    active_after = len([s for s in batch_state.sequences if not s.finished])
                    if active_after < active_before and active_after > 0:
                        with self._lock:
                            self._metrics["stream_batch_shrink_events"] += 1.0

                    # Clear batch if all done
                    if all(s.finished for s in batch_state.sequences):
                        batch_state = None
                        batch_tasks = []

            # Legacy D stage: iterate fallback decode tasks
            if fallback_decode:
                with self._lock:
                    self._metrics["decode_rounds"] += 1.0
                next_decode: List[_ActiveTask] = []
                for state in fallback_decode:
                    status = _step_once(state)
                    if status == "keep":
                        next_decode.append(state)
                fallback_decode = next_decode
