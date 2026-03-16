"""
连续批处理引擎：按迭代为周期的状态机循环。

- SlotManager：管理 KV-Cache 槽位分配与回收。
- RequestState：单个请求在生命周期内的状态。
- Engine：后台 _step_loop 执行 Prefill + Batched Decode，与 FastAPI 通过 out_queue 解耦。
"""
from __future__ import annotations

import queue
import threading
import ctypes
from ctypes import POINTER, cast, c_float, c_int, c_int64, c_size_t, c_ulonglong
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..libllaisys import LIB_LLAISYS

if TYPE_CHECKING:
    from ..models.qwen2 import Qwen2


class _StreamError:
    """流式输出错误标记，放入 out_queue 后消费者应处理并结束流。"""
    def __init__(self, message: str):
        self.message = message


class SlotManager:
    """
    管理 KV-Cache 的槽位分配。
    使用列表存储空闲的 slot_id，确保不会超过 max_batch_size。
    """
    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.free_slots = list(range(max_batch_size))
        self.used_slots: set[int] = set()

    def allocate(self) -> int:
        """弹出并返回一个空闲槽位；无空闲时抛出 RuntimeError。"""
        if not self.free_slots:
            raise RuntimeError("No free slots available")
        slot_id = self.free_slots.pop(0)
        self.used_slots.add(slot_id)
        return slot_id

    def free(self, slot_id: int) -> None:
        """将使用完毕的槽位回收。"""
        if slot_id in self.used_slots:
            self.used_slots.discard(slot_id)
            self.free_slots.append(slot_id)


class RequestState:
    """
    单个请求在生命周期内的状态。
    """
    def __init__(
        self,
        request_id: str,
        prompt_tokens: list[int],
        max_tokens: int,
        out_queue: queue.Queue,
        *,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        seed: int = 0,
        session_id: Optional[str] = None,
        request_messages: Optional[list] = None,
    ):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.out_queue = out_queue

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed

        self.slot_id: int = -1
        self.generated_tokens: list[int] = []
        self.is_finished: bool = False
        self.last_token_id: int = -1

        self.session_id = session_id
        self.request_messages = request_messages


class Engine:
    """
    连续批处理引擎：pending_queue 接收新请求，_step_loop 中 Prefill + Batched Decode，
    结果通过 RequestState.out_queue 回写，与网络 I/O 解耦。
    支持 KV 池：get_kv/put_kv 可选，prefill 前查池命中则 suffix prefill，请求完成后写回池。
    """
    def __init__(
        self,
        model: "Qwen2",
        max_batch_size: int,
        pending_maxsize: int = 64,
        *,
        get_kv: Optional[Callable[..., Any]] = None,
        put_kv: Optional[Callable[..., Any]] = None,
    ):
        self.model = model
        self._c_model = model._model
        self._end_token = model._end_token
        self.max_batch_size = max_batch_size
        self.slot_manager = SlotManager(max_batch_size)
        self._get_kv = get_kv  # (session_id, request_messages, input_ids) -> (prefix_len, blob)
        self._put_kv = put_kv  # (session_id, request_messages, blob, prefix_len) -> None

        self.pending_queue: queue.Queue = queue.Queue(maxsize=pending_maxsize)
        self.running_requests: list[RequestState] = []

        self._engine_thread = threading.Thread(target=self._step_loop, daemon=True)
        self._engine_thread.start()

    def submit_request(self, req_state: RequestState) -> None:
        """供 FastAPI 路由调用，将新请求加入等待队列。队列满时抛出 queue.Full。"""
        self.pending_queue.put_nowait(req_state)

    def _do_prefill(self, req: RequestState) -> int:
        """
        对指定请求做 Prefill：若 KV 池命中则 import 到该 slot 后只对 suffix 做 prefill；
        否则重置该 slot 并传入完整 prompt。返回首个生成的 token_id。
        """
        if not req.prompt_tokens:
            return self._end_token
        input_ids = req.prompt_tokens
        prefix_len = 0
        blob = None
        if self._get_kv and req.session_id and req.request_messages:
            prefix_len, blob = self._get_kv(req.session_id, req.request_messages, input_ids)
        if prefix_len > 0 and blob:
            self.model.import_kv_cache_slot(req.slot_id, blob, prefix_len)
            suffix = input_ids[prefix_len:]
            if not suffix:
                return self._end_token
            n = len(suffix)
            token_arr = (c_int64 * n)(*suffix)
            first_token = LIB_LLAISYS.llaisysQwen2ModelInferWithSlot(
                self._c_model,
                c_size_t(req.slot_id),
                cast(token_arr, POINTER(c_int64)),
                c_size_t(n),
                c_float(req.temperature),
                c_int(req.top_k),
                c_float(req.top_p),
                c_ulonglong(req.seed),
            )
            return int(first_token)
        LIB_LLAISYS.llaisysQwen2ModelResetKVCacheSlot(self._c_model, req.slot_id)
        n = len(input_ids)
        token_arr = (c_int64 * n)(*input_ids)
        first_token = LIB_LLAISYS.llaisysQwen2ModelInferWithSlot(
            self._c_model,
            c_size_t(req.slot_id),
            cast(token_arr, POINTER(c_int64)),
            c_size_t(n),
            c_float(req.temperature),
            c_int(req.top_k),
            c_float(req.top_p),
            c_ulonglong(req.seed),
        )
        return int(first_token)

    def _step_loop(self) -> None:
        """
        引擎主循环：Prefill 新请求 → Batched Decode 当前请求 → 状态更新与槽位回收。
        """
        while True:
            # 阶段 1：Prefill（有空闲槽位且有待处理请求时）
            while self.free_slots and not self.pending_queue.empty():
                try:
                    req = self.pending_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    req.slot_id = self.slot_manager.allocate()
                except RuntimeError:
                    self.pending_queue.put(req)
                    break
                try:
                    first_token = self._do_prefill(req)
                except Exception as e:
                    self.slot_manager.free(req.slot_id)
                    LIB_LLAISYS.llaisysQwen2ModelResetKVCacheSlot(self._c_model, req.slot_id)
                    req.out_queue.put(_StreamError(str(e)))
                    req.out_queue.put(None)
                    continue
                req.last_token_id = first_token
                req.generated_tokens.append(first_token)
                req.out_queue.put(first_token)
                self.running_requests.append(req)

            # 无运行中请求时，阻塞等待新请求以降低 CPU 占用
            if not self.running_requests:
                req = self.pending_queue.get()
                self.pending_queue.put(req)
                continue

            # 阶段 2：Batched Decode
            n_batch = len(self.running_requests)
            slot_ids_array = (c_size_t * n_batch)()
            token_ids_array = (c_int64 * n_batch)()
            out_next_tokens = (c_int64 * n_batch)()

            first = self.running_requests[0]
            for i, r in enumerate(self.running_requests):
                slot_ids_array[i] = r.slot_id
                token_ids_array[i] = r.last_token_id

            LIB_LLAISYS.llaisysQwen2ModelBatchedDecode(
                self._c_model,
                cast(slot_ids_array, POINTER(c_size_t)),
                cast(token_ids_array, POINTER(c_int64)),
                c_size_t(n_batch),
                cast(out_next_tokens, POINTER(c_int64)),
                c_float(first.temperature),
                c_int(first.top_k),
                c_float(first.top_p),
                c_ulonglong(first.seed),
            )

            # 阶段 3：状态更新与清理（含 4.3.5 完成后写回 KV 池）
            active_requests: list[RequestState] = []
            for i, req in enumerate(self.running_requests):
                next_token = int(out_next_tokens[i])
                req.last_token_id = next_token
                req.generated_tokens.append(next_token)
                req.out_queue.put(next_token)

                if next_token == self._end_token or len(req.generated_tokens) >= req.max_tokens:
                    req.is_finished = True
                    req.out_queue.put(None)
                    if self._put_kv and req.session_id and req.request_messages:
                        try:
                            blob = self.model.export_kv_cache_slot(req.slot_id)
                            prefix_len = len(req.prompt_tokens)
                            self._put_kv(req.session_id, req.request_messages, blob, prefix_len)
                        except Exception:
                            pass
                    self.slot_manager.free(req.slot_id)
                    LIB_LLAISYS.llaisysQwen2ModelResetKVCacheSlot(self._c_model, req.slot_id)
                else:
                    active_requests.append(req)

            self.running_requests = active_requests

    @property
    def free_slots(self) -> list[int]:
        """当前空闲槽位列表（只读视图，用于判断是否有空位）。"""
        return self.slot_manager.free_slots

    def get_metrics(self) -> dict:
        """返回引擎监控指标（4.5.3），供 /v1/metrics 等使用。"""
        return {
            "pending_queue_size": self.pending_queue.qsize(),
            "running_count": len(self.running_requests),
            "free_slots_count": len(self.slot_manager.free_slots),
            "max_batch_size": self.max_batch_size,
        }
