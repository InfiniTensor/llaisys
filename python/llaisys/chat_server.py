import argparse
import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

import llaisys
from transformers import AutoTokenizer

try:
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "FastAPI support is optional. Install with `pip install ./python[server]`."
    ) from exc


_STREAM_END = object()


def _device_from_name(device_name: str) -> llaisys.DeviceType:
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    if device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    raise ValueError(f"Unsupported device: {device_name}")


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object")
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if not role:
            raise ValueError("Each message must include a role")
        if isinstance(content, list):
            text_parts = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
            content = "".join(text_parts)
        normalized.append({"role": role, "content": str(content)})
    if not normalized:
        raise ValueError("messages must not be empty")
    return normalized


def _clone_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{"role": message["role"], "content": message["content"]} for message in messages]


def _build_input_ids(tokenizer, messages: List[Dict[str, Any]]) -> List[int]:
    prompt = tokenizer.apply_chat_template(
        conversation=_normalize_messages(messages),
        add_generation_prompt=True,
        tokenize=False,
    )
    return list(tokenizer.encode(prompt))


def _finish_reason(end_token: Optional[int], generated_ids: List[int], max_tokens: int) -> str:
    if generated_ids and generated_ids[-1] == end_token:
        return "stop"
    if len(generated_ids) >= max_tokens:
        return "length"
    return "stop"


def _usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _sse_payload(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _resolve_session_id(payload: Dict[str, Any]) -> str:
    session_id = payload.get("session_id")
    return str(session_id) if session_id else f"session-{uuid.uuid4().hex}"


def _session_title(messages: List[Dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "user" and message["content"]:
            title = message["content"].strip().replace("\n", " ")
            return title[:64] if title else "Untitled Session"
    return "Untitled Session"


def _longest_common_prefix_len(lhs: List[int], rhs: List[int]) -> int:
    limit = min(len(lhs), len(rhs))
    idx = 0
    while idx < limit and lhs[idx] == rhs[idx]:
        idx += 1
    return idx


def _decode_text_delta(tokenizer, generated_ids: List[int], previous_text: str) -> tuple[str, str]:
    decoded = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if decoded.startswith(previous_text):
        return decoded[len(previous_text):], decoded
    return decoded, decoded


def _model_generate_next(
    model: Any,
    inputs: List[int],
    *,
    top_k: int,
    top_p: float,
    temperature: float,
    reset_state: bool,
) -> int:
    generate_next = getattr(model, "generate_next", None)
    if callable(generate_next):
        return int(
            generate_next(
                inputs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                reset_state=reset_state,
            )
        )

    if reset_state and hasattr(model, "reset"):
        model.reset()

    private_generate_next = getattr(model, "_generate_next", None)
    if callable(private_generate_next):
        return int(
            private_generate_next(
                inputs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        )

    output_ids = model.generate(
        inputs,
        max_new_tokens=1,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        reset_state=reset_state,
    )
    return int(output_ids[-1])


@dataclass
class SessionState:
    session_id: str
    messages: List[Dict[str, str]]
    token_ids: List[int]
    created_at: float
    updated_at: float
    worker_id: Optional[int] = None


@dataclass
class ReusePlan:
    reset_state: bool
    generation_inputs: List[int]
    reused_tokens: int
    truncate_to: Optional[int]


@dataclass
class CompletionRequest:
    completion_id: str
    session_id: str
    messages: List[Dict[str, str]]
    input_ids: List[int]
    max_tokens: int
    top_k: int
    top_p: float
    temperature: float
    stream: bool
    created: int
    request_model: str
    result_queue: "queue.Queue[Any]"
    enqueued_at: float
    batch_id: Optional[int] = None
    first_batch_id: Optional[int] = None
    generated_ids: List[int] = field(default_factory=list)
    last_text: str = ""
    cache_reused_tokens: int = 0
    started: bool = False
    prompt_inputs: List[int] = field(default_factory=list)
    reset_state: bool = True
    steps_dispatched: int = 0
    assigned_worker_id: Optional[int] = None
    first_dispatched_at: Optional[float] = None
    finished_at: Optional[float] = None


@dataclass
class WorkerState:
    worker_id: int
    model: Any
    job_queue: "queue.Queue[Optional[CompletionRequest]]" = field(default_factory=queue.Queue)
    thread: Optional[threading.Thread] = None
    busy: bool = False
    current_request_id: Optional[str] = None
    active_session_id: Optional[str] = None
    active_token_ids: List[int] = field(default_factory=list)
    needs_reset: bool = True
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_steps: int = 0
    last_used_at: float = 0.0


@dataclass
class ServiceState:
    tokenizer: Any
    model_name: str
    workers: List[WorkerState]
    max_batch_size: int
    batch_wait_s: float
    sessions: Dict[str, SessionState] = field(default_factory=dict)
    session_inflight: Dict[str, int] = field(default_factory=dict)
    pending_requests: Deque[CompletionRequest] = field(default_factory=deque)
    condition: threading.Condition = field(default_factory=threading.Condition)
    shutdown: bool = False
    next_batch_id: int = 1
    scheduled_batches: int = 0
    max_observed_batch_size: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_enqueued: int = 0
    total_generated_tokens: int = 0
    requeued_requests: int = 0
    cache_reuse_hits: int = 0
    total_queue_wait_s: float = 0.0
    total_request_time_s: float = 0.0
    scheduler_thread: Optional[threading.Thread] = None


def _session_payload(
    state: SessionState,
    *,
    include_messages: bool = False,
    inflight: int = 0,
) -> Dict[str, Any]:
    payload = {
        "id": state.session_id,
        "object": "session",
        "title": _session_title(state.messages),
        "created": int(state.created_at),
        "updated": int(state.updated_at),
        "message_count": len(state.messages),
        "token_count": len(state.token_ids),
        "worker_id": state.worker_id,
        "inflight": inflight,
    }
    if include_messages:
        payload["messages"] = _clone_messages(state.messages)
    return payload


def _prepare_worker_reuse(worker: WorkerState, session_id: str, input_ids: List[int]) -> ReusePlan:
    if worker.needs_reset or worker.active_session_id != session_id or not worker.active_token_ids:
        return ReusePlan(reset_state=True, generation_inputs=input_ids, reused_tokens=0, truncate_to=None)

    prefix_len = _longest_common_prefix_len(worker.active_token_ids, input_ids)
    if prefix_len <= 0:
        return ReusePlan(reset_state=True, generation_inputs=input_ids, reused_tokens=0, truncate_to=None)

    reusable_prefix = prefix_len
    if reusable_prefix >= len(input_ids):
        reusable_prefix = len(input_ids) - 1

    if reusable_prefix <= 0:
        return ReusePlan(reset_state=True, generation_inputs=input_ids, reused_tokens=0, truncate_to=None)

    truncate_to = reusable_prefix if reusable_prefix < len(worker.active_token_ids) else None
    return ReusePlan(
        reset_state=False,
        generation_inputs=input_ids[reusable_prefix:],
        reused_tokens=reusable_prefix,
        truncate_to=truncate_to,
    )


def _detach_worker_session_locked(service: ServiceState, worker: WorkerState) -> None:
    if worker.active_session_id is None:
        worker.active_token_ids = []
        worker.needs_reset = True
        return

    state = service.sessions.get(worker.active_session_id)
    if state is not None and state.worker_id == worker.worker_id:
        state.worker_id = None
    worker.active_session_id = None
    worker.active_token_ids = []
    worker.needs_reset = True


def _clear_session_cache_locked(service: ServiceState, session_id: str) -> None:
    state = service.sessions.get(session_id)
    if state is None:
        return
    worker_id = state.worker_id
    state.worker_id = None
    state.token_ids = []
    if worker_id is None:
        return
    worker = service.workers[worker_id]
    if worker.active_session_id == session_id:
        worker.active_session_id = None
        worker.active_token_ids = []
        worker.needs_reset = True


def _decrement_inflight_locked(service: ServiceState, session_id: str) -> None:
    remaining = service.session_inflight.get(session_id, 0) - 1
    if remaining > 0:
        service.session_inflight[session_id] = remaining
    else:
        service.session_inflight.pop(session_id, None)


def _commit_session_locked(
    service: ServiceState,
    worker: WorkerState,
    session_id: str,
    messages: List[Dict[str, str]],
    cached_tokens: List[int],
) -> None:
    now = time.time()
    previous = service.sessions.get(session_id)
    created_at = previous.created_at if previous is not None else now
    service.sessions[session_id] = SessionState(
        session_id=session_id,
        messages=_clone_messages(messages),
        token_ids=list(cached_tokens),
        created_at=created_at,
        updated_at=now,
        worker_id=worker.worker_id,
    )
    worker.active_session_id = session_id
    worker.active_token_ids = list(cached_tokens)
    worker.needs_reset = False


def _select_worker_locked(
    service: ServiceState,
    request: CompletionRequest,
    idle_workers: Dict[int, WorkerState],
    pinned_session_ids: set[str],
) -> Optional[WorkerState]:
    session = service.sessions.get(request.session_id)
    if session is not None and session.worker_id in idle_workers:
        return idle_workers[session.worker_id]

    for worker in idle_workers.values():
        if worker.active_session_id == request.session_id and not worker.needs_reset:
            return worker

    never_used = [worker for worker in idle_workers.values() if worker.active_session_id is None]
    if never_used:
        return min(never_used, key=lambda item: item.last_used_at)

    reusable_workers = [
        worker
        for worker in idle_workers.values()
        if worker.active_session_id not in pinned_session_ids
    ]
    if reusable_workers:
        return min(reusable_workers, key=lambda item: item.last_used_at)

    return min(idle_workers.values(), key=lambda item: item.last_used_at, default=None)


def _service_stats(service: ServiceState) -> Dict[str, Any]:
    with service.condition:
        workers = [
            {
                "worker_id": worker.worker_id,
                "busy": worker.busy,
                "active_session_id": worker.active_session_id,
                "cached_tokens": len(worker.active_token_ids),
                "needs_reset": worker.needs_reset,
                "total_requests": worker.total_requests,
                "total_tokens_generated": worker.total_tokens_generated,
                "total_steps": worker.total_steps,
            }
            for worker in service.workers
        ]
        return {
            "object": "service.stats",
            "model": service.model_name,
            "worker_count": len(service.workers),
            "max_batch_size": service.max_batch_size,
            "batch_wait_ms": int(service.batch_wait_s * 1000),
            "queue_depth": len(service.pending_requests),
            "active_requests": sum(1 for worker in service.workers if worker.busy),
            "session_count": len(service.sessions),
            "scheduled_batches": service.scheduled_batches,
            "max_observed_batch_size": service.max_observed_batch_size,
            "completed_requests": service.completed_requests,
            "failed_requests": service.failed_requests,
            "total_enqueued": service.total_enqueued,
            "total_generated_tokens": service.total_generated_tokens,
            "requeued_requests": service.requeued_requests,
            "cache_reuse_hits": service.cache_reuse_hits,
            "avg_queue_wait_ms": (
                (service.total_queue_wait_s / service.completed_requests) * 1000.0
                if service.completed_requests
                else 0.0
            ),
            "avg_request_time_ms": (
                (service.total_request_time_s / service.completed_requests) * 1000.0
                if service.completed_requests
                else 0.0
            ),
            "cached_session_count": sum(1 for session in service.sessions.values() if session.worker_id is not None),
            "workers": workers,
        }


def _scheduler_loop(service: ServiceState) -> None:
    while True:
        assigned: List[tuple[WorkerState, CompletionRequest]] = []
        with service.condition:
            while not service.shutdown and not service.pending_requests:
                service.condition.wait()
            if service.shutdown:
                return

            if len(service.pending_requests) < service.max_batch_size and service.batch_wait_s > 0:
                service.condition.wait(timeout=service.batch_wait_s)
                if service.shutdown:
                    return

            candidates: List[CompletionRequest] = []
            while service.pending_requests and len(candidates) < service.max_batch_size:
                candidates.append(service.pending_requests.popleft())

            idle_workers = {
                worker.worker_id: worker
                for worker in service.workers
                if not worker.busy
            }
            pinned_session_ids = {
                pending_request.session_id
                for pending_request in service.pending_requests
                if pending_request.started
            }
            pinned_session_ids.update(
                candidate.session_id
                for candidate in candidates
                if candidate.started
            )
            unassigned: List[CompletionRequest] = []
            if idle_workers:
                batch_id = service.next_batch_id
                service.next_batch_id += 1
                for request in candidates:
                    worker = _select_worker_locked(service, request, idle_workers, pinned_session_ids)
                    if worker is None:
                        unassigned.append(request)
                        continue
                    if worker.active_session_id is not None and worker.active_session_id != request.session_id:
                        _detach_worker_session_locked(service, worker)
                    worker.busy = True
                    worker.current_request_id = request.completion_id
                    request.batch_id = batch_id
                    if request.first_batch_id is None:
                        request.first_batch_id = batch_id
                    request.assigned_worker_id = worker.worker_id
                    request.steps_dispatched += 1
                    if request.first_dispatched_at is None:
                        request.first_dispatched_at = time.time()
                    assigned.append((worker, request))
                    idle_workers.pop(worker.worker_id, None)
                    pinned_session_ids.add(request.session_id)

                if assigned:
                    service.scheduled_batches += 1
                    service.max_observed_batch_size = max(
                        service.max_observed_batch_size,
                        len(assigned),
                    )
            else:
                for request in reversed(candidates):
                    service.pending_requests.appendleft(request)
                service.condition.wait(timeout=max(service.batch_wait_s, 0.01))
                continue

            for request in reversed(unassigned):
                service.pending_requests.appendleft(request)

        for worker, request in assigned:
            worker.job_queue.put(request)


def _worker_loop(service: ServiceState, worker: WorkerState) -> None:
    while True:
        request = worker.job_queue.get()
        if request is None:
            return
        _run_request(service, worker, request)


def _run_request(service: ServiceState, worker: WorkerState, request: CompletionRequest) -> None:
    try:
        emit_role_chunk = False
        with service.condition:
            batch_id = request.batch_id
            worker_id = worker.worker_id
            truncate_to = None

            if not request.started:
                plan = _prepare_worker_reuse(worker, request.session_id, request.input_ids)
                request.cache_reused_tokens = plan.reused_tokens
                request.prompt_inputs = list(plan.generation_inputs)
                request.reset_state = plan.reset_state
                request.started = True
                request.assigned_worker_id = worker_id
                truncate_to = plan.truncate_to
                if request.cache_reused_tokens > 0:
                    service.cache_reuse_hits += 1
                emit_role_chunk = request.stream

        if truncate_to is not None:
            worker.model.truncate(truncate_to)

        if emit_role_chunk:
            request.result_queue.put(
                _sse_payload(
                    {
                        "id": request.completion_id,
                        "object": "chat.completion.chunk",
                        "created": request.created,
                        "model": request.request_model,
                        "session_id": request.session_id,
                        "worker_id": worker_id,
                        "batch_id": batch_id,
                        "cache_reused_tokens": request.cache_reused_tokens,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )
            )

        token_source = request.prompt_inputs if not request.generated_ids else [request.generated_ids[-1]]
        next_token = _model_generate_next(
            worker.model,
            token_source,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            reset_state=request.reset_state if not request.generated_ids else False,
        )
        request.generated_ids.append(next_token)
        text_chunk, request.last_text = _decode_text_delta(
            service.tokenizer,
            request.generated_ids,
            request.last_text,
        )

        finished = (
            next_token == getattr(worker.model.meta, "end_token", None)
            or len(request.generated_ids) >= request.max_tokens
        )
        assistant_content = service.tokenizer.decode(
            request.generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        finish_reason = _finish_reason(
            getattr(worker.model.meta, "end_token", None),
            request.generated_ids,
            request.max_tokens,
        )

        with service.condition:
            worker.busy = False
            worker.current_request_id = None
            worker.last_used_at = time.time()
            worker.total_steps += 1
            worker.total_tokens_generated += 1
            service.total_generated_tokens += 1

            if finished:
                request.finished_at = time.time()
                _commit_session_locked(
                    service,
                    worker,
                    request.session_id,
                    request.messages + [{"role": "assistant", "content": assistant_content}],
                    request.input_ids + request.generated_ids,
                )
                worker.total_requests += 1
                service.completed_requests += 1
                if request.first_dispatched_at is not None:
                    service.total_queue_wait_s += max(0.0, request.first_dispatched_at - request.enqueued_at)
                    service.total_request_time_s += max(0.0, request.finished_at - request.enqueued_at)
                _decrement_inflight_locked(service, request.session_id)
            else:
                worker.active_session_id = request.session_id
                worker.active_token_ids = list(request.input_ids + request.generated_ids)
                worker.needs_reset = False
                service.requeued_requests += 1
                service.pending_requests.append(request)
                service.condition.notify_all()
            service.condition.notify_all()

        if request.stream and text_chunk:
            request.result_queue.put(
                _sse_payload(
                    {
                        "id": request.completion_id,
                        "object": "chat.completion.chunk",
                        "created": request.created,
                        "model": request.request_model,
                        "session_id": request.session_id,
                        "worker_id": worker_id,
                        "batch_id": batch_id,
                        "cache_reused_tokens": request.cache_reused_tokens,
                        "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}],
                    }
                )
            )

        if not finished:
            return

        if request.stream:
            request.result_queue.put(
                _sse_payload(
                    {
                        "id": request.completion_id,
                        "object": "chat.completion.chunk",
                        "created": request.created,
                        "model": request.request_model,
                        "session_id": request.session_id,
                        "worker_id": worker_id,
                        "batch_id": request.first_batch_id,
                        "last_batch_id": batch_id,
                        "dispatch_count": request.steps_dispatched,
                        "cache_reused_tokens": request.cache_reused_tokens,
                        "queue_wait_ms": (
                            max(0.0, (request.first_dispatched_at or request.enqueued_at) - request.enqueued_at) * 1000.0
                        ),
                        "total_time_ms": (
                            max(0.0, (request.finished_at or time.time()) - request.enqueued_at) * 1000.0
                        ),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    }
                )
            )
            request.result_queue.put("data: [DONE]\n\n")
            request.result_queue.put(_STREAM_END)
            return

        request.result_queue.put(
            {
                "id": request.completion_id,
                "object": "chat.completion",
                "created": request.created,
                "model": request.request_model,
                "session_id": request.session_id,
                "worker_id": worker_id,
                "batch_id": request.first_batch_id,
                "last_batch_id": batch_id,
                "dispatch_count": request.steps_dispatched,
                "cache_reused_tokens": request.cache_reused_tokens,
                "queue_wait_ms": (
                    max(0.0, (request.first_dispatched_at or request.enqueued_at) - request.enqueued_at) * 1000.0
                ),
                "total_time_ms": (
                    max(0.0, (request.finished_at or time.time()) - request.enqueued_at) * 1000.0
                ),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": assistant_content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": _usage(len(request.input_ids), len(request.generated_ids)),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive path
        with service.condition:
            _detach_worker_session_locked(service, worker)
            worker.busy = False
            worker.current_request_id = None
            worker.last_used_at = time.time()
            service.failed_requests += 1
            _decrement_inflight_locked(service, request.session_id)
            service.condition.notify_all()

        if request.stream:
            request.result_queue.put(
                _sse_payload(
                    {
                        "id": request.completion_id,
                        "object": "chat.completion.chunk",
                        "created": request.created,
                        "model": request.request_model,
                        "session_id": request.session_id,
                        "worker_id": worker.worker_id,
                        "batch_id": request.batch_id,
                        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
                        "error": str(exc),
                    }
                )
            )
            request.result_queue.put("data: [DONE]\n\n")
            request.result_queue.put(_STREAM_END)
            return

        request.result_queue.put(exc)


def _shutdown_service(service: ServiceState) -> None:
    with service.condition:
        if service.shutdown:
            return
        service.shutdown = True
        service.condition.notify_all()

    if service.scheduler_thread is not None:
        service.scheduler_thread.join(timeout=5)
        service.scheduler_thread = None

    for worker in service.workers:
        worker.job_queue.put(None)
    for worker in service.workers:
        if worker.thread is not None:
            worker.thread.join(timeout=5)
            worker.thread = None


def _resolve_runtime_components(
    model_path: Optional[str],
    device: str,
    device_id: int,
    tp_size: int,
    tp_device_ids: Optional[List[int]],
    *,
    tokenizer,
    model,
    model_factory,
    model_name: Optional[str],
    num_workers: int,
) -> tuple[Any, Callable[[], Any], str]:
    if tokenizer is None:
        if model_path is None:
            raise ValueError("model_path is required when tokenizer is not injected")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if model is not None and model_factory is not None:
        raise ValueError("Pass either `model` or `model_factory`, not both")

    if model_factory is None:
        if model is not None:
            if num_workers != 1:
                raise ValueError("model_factory is required when injecting a model with num_workers > 1")
            model_factory = lambda: model
            resolved_model_name = model_name or "llaisys-chat"
            return tokenizer, model_factory, resolved_model_name

        if model_path is None:
            raise ValueError("model_path is required when model/model_factory are not injected")

        device_type = _device_from_name(device)

        def model_factory() -> Any:
            return llaisys.models.create_model(
                model_path,
                device_type,
                device_id,
                tp_size=tp_size,
                tp_device_ids=tp_device_ids,
            )

        resolved_model_name = model_name or llaisys.models.default_model_name(model_path)
        return tokenizer, model_factory, resolved_model_name

    resolved_model_name = model_name or "llaisys-chat"
    return tokenizer, model_factory, resolved_model_name


def create_app(
    model_path: Optional[str] = None,
    device: str = "cpu",
    device_id: int = 0,
    tp_size: int = 1,
    tp_device_ids: Optional[List[int]] = None,
    *,
    tokenizer=None,
    model=None,
    model_factory: Optional[Callable[[], Any]] = None,
    model_name: Optional[str] = None,
    num_workers: int = 1,
    max_batch_size: int = 1,
    batch_wait_ms: int = 5,
) -> FastAPI:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if batch_wait_ms < 0:
        raise ValueError("batch_wait_ms must be non-negative")
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if tp_size > 1 and device != "nvidia":
        raise ValueError("Tensor parallel service mode currently supports only NVIDIA devices")
    if tp_size > 1 and num_workers != 1:
        raise ValueError("tp_size > 1 currently requires num_workers == 1")

    tokenizer, model_factory, resolved_model_name = _resolve_runtime_components(
        model_path,
        device,
        device_id,
        tp_size,
        tp_device_ids,
        tokenizer=tokenizer,
        model=model,
        model_factory=model_factory,
        model_name=model_name,
        num_workers=num_workers,
    )

    workers = [WorkerState(worker_id=i, model=model_factory()) for i in range(num_workers)]
    service = ServiceState(
        tokenizer=tokenizer,
        model_name=resolved_model_name,
        workers=workers,
        max_batch_size=max_batch_size,
        batch_wait_s=batch_wait_ms / 1000.0,
    )

    for worker in service.workers:
        worker.thread = threading.Thread(
            target=_worker_loop,
            args=(service, worker),
            name=f"llaisys-worker-{worker.worker_id}",
            daemon=True,
        )
        worker.thread.start()

    service.scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        args=(service,),
        name="llaisys-scheduler",
        daemon=True,
    )
    service.scheduler_thread.start()

    app = FastAPI(title="LLAISYS Chat API", version="0.3.0")
    app.state.service = service
    app.state.tokenizer = tokenizer
    app.state.model_name = resolved_model_name

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        _shutdown_service(app.state.service)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        stats = _service_stats(app.state.service)
        return {
            "status": "ok",
            "model": app.state.model_name,
            "sessions": stats["session_count"],
            "queue_depth": stats["queue_depth"],
            "active_requests": stats["active_requests"],
            "worker_count": stats["worker_count"],
            "tp_size": tp_size,
            "tp_device_ids": list(tp_device_ids or []),
        }

    @app.get("/v1/service/stats")
    def service_stats() -> Dict[str, Any]:
        return _service_stats(app.state.service)

    @app.get("/v1/sessions")
    def list_sessions() -> Dict[str, Any]:
        service = app.state.service
        with service.condition:
            sessions = sorted(service.sessions.values(), key=lambda item: item.updated_at, reverse=True)
            return {
                "object": "list",
                "data": [
                    _session_payload(
                        session,
                        inflight=service.session_inflight.get(session.session_id, 0),
                    )
                    for session in sessions
                ],
            }

    @app.post("/v1/sessions")
    def create_session(payload: Optional[Dict[str, Any]] = Body(default=None)) -> JSONResponse:
        payload = payload or {}
        session_id = _resolve_session_id(payload)
        messages = payload.get("messages")
        normalized_messages = _normalize_messages(messages) if messages else []
        service = app.state.service

        with service.condition:
            if session_id in service.sessions:
                raise HTTPException(status_code=409, detail=f"Session `{session_id}` already exists")

            now = time.time()
            service.sessions[session_id] = SessionState(
                session_id=session_id,
                messages=normalized_messages,
                token_ids=[],
                created_at=now,
                updated_at=now,
            )
            return JSONResponse(_session_payload(service.sessions[session_id], include_messages=True))

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str) -> Dict[str, Any]:
        service = app.state.service
        with service.condition:
            state = service.sessions.get(session_id)
            if state is None:
                raise HTTPException(status_code=404, detail=f"Unknown session `{session_id}`")
            payload = _session_payload(
                state,
                include_messages=True,
                inflight=service.session_inflight.get(session_id, 0),
            )
            payload["active"] = state.worker_id is not None
            return payload

    @app.put("/v1/sessions/{session_id}")
    def replace_session(session_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if "messages" not in payload:
            raise HTTPException(status_code=400, detail="`messages` is required")

        try:
            normalized_messages = _normalize_messages(payload["messages"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        service = app.state.service
        with service.condition:
            state = service.sessions.get(session_id)
            if state is None:
                raise HTTPException(status_code=404, detail=f"Unknown session `{session_id}`")
            if service.session_inflight.get(session_id, 0) > 0:
                raise HTTPException(status_code=409, detail=f"Session `{session_id}` is busy")

            _clear_session_cache_locked(service, session_id)
            state.messages = normalized_messages
            state.updated_at = time.time()
            return _session_payload(state, include_messages=True)

    @app.delete("/v1/sessions/{session_id}")
    def delete_session(session_id: str) -> Dict[str, Any]:
        service = app.state.service
        with service.condition:
            state = service.sessions.get(session_id)
            if state is None:
                raise HTTPException(status_code=404, detail=f"Unknown session `{session_id}`")
            if service.session_inflight.get(session_id, 0) > 0:
                raise HTTPException(status_code=409, detail=f"Session `{session_id}` is busy")

            _clear_session_cache_locked(service, session_id)
            service.sessions.pop(session_id, None)
            return {"id": session_id, "object": "session.deleted", "deleted": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: Dict[str, Any] = Body(...)):
        service = app.state.service
        session_id = _resolve_session_id(payload)
        with service.condition:
            stored_session = service.sessions.get(session_id)

        raw_messages = payload.get("messages")
        if raw_messages is None:
            if stored_session is None:
                raise HTTPException(status_code=400, detail="`messages` is required for a new session")
            raw_messages = stored_session.messages

        try:
            messages = _normalize_messages(raw_messages)
            input_ids = _build_input_ids(service.tokenizer, messages)
            max_tokens = int(payload.get("max_tokens", payload.get("max_completion_tokens", 128)))
            top_k = int(payload.get("top_k", 1))
            top_p = float(payload.get("top_p", 1.0))
            temperature = float(payload.get("temperature", 1.0))
            stream = bool(payload.get("stream", False))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if max_tokens <= 0:
            raise HTTPException(status_code=400, detail="`max_tokens` must be positive")

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        request_model = str(payload.get("model", app.state.model_name))
        result_queue: "queue.Queue[Any]" = queue.Queue()
        request = CompletionRequest(
            completion_id=completion_id,
            session_id=session_id,
            messages=messages,
            input_ids=input_ids,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stream=stream,
            created=created,
            request_model=request_model,
            result_queue=result_queue,
            enqueued_at=time.time(),
        )

        with service.condition:
            if service.session_inflight.get(session_id, 0) > 0:
                raise HTTPException(status_code=409, detail=f"Session `{session_id}` already has an in-flight request")
            service.session_inflight[session_id] = service.session_inflight.get(session_id, 0) + 1
            service.pending_requests.append(request)
            service.total_enqueued += 1
            service.condition.notify_all()

        if stream:
            def event_stream() -> Iterable[str]:
                while True:
                    item = result_queue.get()
                    if item is _STREAM_END:
                        break
                    yield item

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        result = result_queue.get()
        if isinstance(result, Exception):
            raise HTTPException(status_code=500, detail=str(result))
        return JSONResponse(result)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model-name", default=None, type=str)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--max-batch-size", default=1, type=int)
    parser.add_argument("--batch-wait-ms", default=5, type=int)
    parser.add_argument("--tp-size", default=1, type=int)
    parser.add_argument("--tp-device-ids", default=None, type=str)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Uvicorn is optional. Install with `pip install ./python[server]`."
        ) from exc

    tp_device_ids = None
    if args.tp_device_ids:
        tp_device_ids = [int(part.strip()) for part in args.tp_device_ids.split(",") if part.strip()]

    app = create_app(
        model_path=args.model_path,
        device=args.device,
        device_id=args.device_id,
        tp_size=args.tp_size,
        tp_device_ids=tp_device_ids,
        model_name=args.model_name,
        num_workers=args.num_workers,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
