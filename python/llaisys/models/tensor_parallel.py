import ctypes
import json
import mmap
import os
import queue
import socket
import struct
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..libllaisys import DataType, DeviceType, MemcpyKind
from ..ops import Ops
from ..runtime import RuntimeAPI
from ..tensor import Tensor
from .qwen2 import Qwen2


def _tp_log(rank: int, message: str) -> None:
    if os.getenv("LLAISYS_TP_DEBUG", "").lower() not in {"1", "true", "yes", "on"}:
        return
    with open(f"/tmp/llaisys_tp_rank{rank}.log", "a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


@dataclass
class _TensorParallelMeta:
    dtype: DataType
    nlayer: int
    hs: int
    nh: int
    nkvh: int
    dh: int
    di: int
    maxseq: int
    voc: int
    epsilon: float
    theta: float
    end_token: int


def _divisors(value: int) -> List[int]:
    divisors: List[int] = []
    for candidate in range(1, int(math.isqrt(value)) + 1):
        if value % candidate != 0:
            continue
        divisors.append(candidate)
        paired = value // candidate
        if paired != candidate:
            divisors.append(paired)
    return sorted(divisors)


def _valid_tp_sizes(meta: _TensorParallelMeta) -> List[int]:
    common = math.gcd(math.gcd(int(meta.nh), int(meta.nkvh)), int(meta.di))
    return _divisors(common)


def _validate_tp_size(meta: _TensorParallelMeta, tp_size: int) -> None:
    tp_size = int(tp_size)
    valid_sizes = _valid_tp_sizes(meta)
    if tp_size in valid_sizes:
        return
    raise ValueError(
        "Unsupported tp_size for current tensor-parallel implementation: "
        f"tp_size={tp_size}, num_attention_heads={meta.nh}, "
        f"num_key_value_heads={meta.nkvh}, intermediate_size={meta.di}, "
        f"valid_tp_sizes={valid_sizes}. "
        "This implementation shards attention heads, KV heads, and MLP intermediate dims evenly across ranks."
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _llaisys_dtype_to_torch(dtype: DataType) -> torch.dtype:
    if dtype == DataType.F32:
        return torch.float32
    if dtype == DataType.F16:
        return torch.float16
    if dtype == DataType.BF16:
        return torch.bfloat16
    raise ValueError(f"Unsupported runtime dtype: {dtype}")


def _resolve_tp_device_ids(device_id: int, tp_size: int, tp_device_ids: Optional[Sequence[int]]) -> List[int]:
    if tp_device_ids is not None:
        device_ids = [int(item) for item in tp_device_ids]
    else:
        device_ids = list(range(int(device_id), int(device_id) + int(tp_size)))
    if len(device_ids) != int(tp_size):
        raise ValueError("tp_device_ids length must match tp_size")
    return device_ids


def _normalize_command_payload(command: str, seq: int, **payload) -> Dict[str, object]:
    message = {"command": command, "seq": int(seq)}
    message.update(payload)
    return message


class _ShardedQwen2Rank:
    def __init__(self, model_path: str, rank: int, world_size: int, device_id: int):
        self.model_path = Path(model_path)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device_id = int(device_id)
        self.device = DeviceType.NVIDIA
        self.runtime = RuntimeAPI(DeviceType.NVIDIA)
        self.runtime.set_device(self.device_id)
        torch.cuda.set_device(self.device_id)

        with open(self.model_path / "config.json", "r") as handle:
            config = json.load(handle)
        self.config = config
        if str(config.get("model_type", "")).lower() != "qwen2":
            raise ValueError("TensorParallelQwen2 currently supports only Qwen2 models")

        dtype = Qwen2._runtime_dtype(str(config.get("torch_dtype", "float32")), DeviceType.NVIDIA)
        eos_token = config.get("eos_token_id", 151643)
        if isinstance(eos_token, list):
            eos_token = eos_token[0] if eos_token else 151643
        self.meta = _TensorParallelMeta(
            dtype=dtype,
            nlayer=int(config.get("num_hidden_layers", 24)),
            hs=int(config.get("hidden_size", 2048)),
            nh=int(config.get("num_attention_heads", 16)),
            nkvh=int(config.get("num_key_value_heads", int(config.get("num_attention_heads", 16)))),
            dh=int(config.get("hidden_size", 2048)) // int(config.get("num_attention_heads", 16)),
            di=int(config.get("intermediate_size", 11008)),
            maxseq=int(config.get("max_position_embeddings", 8192)),
            voc=int(config.get("vocab_size", 151936)),
            epsilon=float(config.get("rms_norm_eps", 1e-6)),
            theta=float(config.get("rope_theta", 1000000.0)),
            end_token=int(eos_token),
        )

        _validate_tp_size(self.meta, self.world_size)

        self.local_nh = self.meta.nh // self.world_size
        self.local_nkvh = self.meta.nkvh // self.world_size
        self.local_q_dim = self.local_nh * self.meta.dh
        self.local_kv_dim = self.local_nkvh * self.meta.dh
        self.local_di = self.meta.di // self.world_size
        self.torch_dtype = _llaisys_dtype_to_torch(self.meta.dtype)
        self.cuda_device = torch.device(f"cuda:{self.device_id}")

        self.in_embed: Optional[Tensor] = None
        self.out_embed: Optional[Tensor] = None
        self.out_norm_w: Optional[Tensor] = None
        self.layers: List[Dict[str, Tensor]] = [dict() for _ in range(self.meta.nlayer)]
        self._tensors: List[Tensor] = []
        self._load_weights()

        if self.out_embed is None and self.config.get("tie_word_embeddings") and self.in_embed is not None:
            self.out_embed = self.in_embed
        if self.in_embed is None or self.out_embed is None or self.out_norm_w is None:
            raise ValueError("Missing required embedding/final norm weights for tensor-parallel Qwen2")

        self.k_caches = [
            Tensor((self.meta.maxseq, self.local_nkvh, self.meta.dh), self.meta.dtype, self.device, self.device_id)
            for _ in range(self.meta.nlayer)
        ]
        self.v_caches = [
            Tensor((self.meta.maxseq, self.local_nkvh, self.meta.dh), self.meta.dtype, self.device, self.device_id)
            for _ in range(self.meta.nlayer)
        ]
        self._cur_pos = 0

    def reset(self) -> None:
        self._cur_pos = 0

    def truncate(self, position: int) -> None:
        position = int(position)
        if position < 0 or position > self._cur_pos:
            raise ValueError("truncate position exceeds current cache length")
        self._cur_pos = position

    def generate_next(self, inputs: Sequence[int], top_k: int, top_p: float, temperature: float) -> int:
        if not inputs:
            raise ValueError("inputs must not be empty")
        if self._cur_pos + len(inputs) > self.meta.maxseq:
            raise ValueError("sequence exceeds KV-cache capacity")

        _tp_log(self.rank, f"generate_next start cur_pos={self._cur_pos} ntoken={len(inputs)}")
        hidden = self._embedding(list(inputs))
        pos_ids = self._make_int64_tensor(np.arange(self._cur_pos, self._cur_pos + len(inputs), dtype=np.int64))
        scale = 1.0 / float(self.meta.dh) ** 0.5

        for layer_idx, layer in enumerate(self.layers):
            _tp_log(self.rank, f"layer {layer_idx} start")
            normed = Tensor(hidden.shape(), self.meta.dtype, self.device, self.device_id)
            Ops.rms_norm(normed, hidden, layer["attn_norm_w"], self.meta.epsilon)
            self._debug_sync(f"layer {layer_idx} attn_norm")

            q = Tensor((len(inputs), self.local_q_dim), self.meta.dtype, self.device, self.device_id)
            k = Tensor((len(inputs), self.local_kv_dim), self.meta.dtype, self.device, self.device_id)
            v = Tensor((len(inputs), self.local_kv_dim), self.meta.dtype, self.device, self.device_id)
            Ops.linear(q, normed, layer["attn_q_w"], layer["attn_q_b"])
            self._debug_sync(f"layer {layer_idx} attn_q")
            Ops.linear(k, normed, layer["attn_k_w"], layer["attn_k_b"])
            self._debug_sync(f"layer {layer_idx} attn_k")
            Ops.linear(v, normed, layer["attn_v_w"], layer["attn_v_b"])
            self._debug_sync(f"layer {layer_idx} attn_v")

            q_heads = q.view(len(inputs), self.local_nh, self.meta.dh)
            k_heads = k.view(len(inputs), self.local_nkvh, self.meta.dh)
            v_heads = v.view(len(inputs), self.local_nkvh, self.meta.dh)
            Ops.rope(q_heads, q_heads, pos_ids, self.meta.theta)
            self._debug_sync(f"layer {layer_idx} rope_q")
            Ops.rope(k_heads, k_heads, pos_ids, self.meta.theta)
            self._debug_sync(f"layer {layer_idx} rope_k")

            k_slot = self.k_caches[layer_idx].slice(0, self._cur_pos, self._cur_pos + len(inputs))
            v_slot = self.v_caches[layer_idx].slice(0, self._cur_pos, self._cur_pos + len(inputs))
            Ops.rearrange(k_slot, k_heads)
            self._debug_sync(f"layer {layer_idx} cache_k")
            Ops.rearrange(v_slot, v_heads)
            self._debug_sync(f"layer {layer_idx} cache_v")

            k_full = self.k_caches[layer_idx].slice(0, 0, self._cur_pos + len(inputs))
            v_full = self.v_caches[layer_idx].slice(0, 0, self._cur_pos + len(inputs))
            attn_local = Tensor((len(inputs), self.local_nh, self.meta.dh), self.meta.dtype, self.device, self.device_id)
            Ops.self_attention(attn_local, q_heads, k_full, v_full, scale)
            self._debug_sync(f"layer {layer_idx} self_attention")

            attn_flat = attn_local.view(len(inputs), self.local_q_dim)
            attn_partial = Tensor((len(inputs), self.meta.hs), self.meta.dtype, self.device, self.device_id)
            Ops.linear(attn_partial, attn_flat, layer["attn_o_w"], None)
            self._debug_sync(f"layer {layer_idx} attn_o")
            self._all_reduce_tensor(attn_partial)
            self._debug_sync(f"layer {layer_idx} attn_all_reduce")
            Ops.add(hidden, hidden, attn_partial)
            self._debug_sync(f"layer {layer_idx} attn_residual")

            mlp_normed = Tensor(hidden.shape(), self.meta.dtype, self.device, self.device_id)
            Ops.rms_norm(mlp_normed, hidden, layer["mlp_norm_w"], self.meta.epsilon)
            self._debug_sync(f"layer {layer_idx} mlp_norm")
            gate = Tensor((len(inputs), self.local_di), self.meta.dtype, self.device, self.device_id)
            up = Tensor((len(inputs), self.local_di), self.meta.dtype, self.device, self.device_id)
            Ops.linear(gate, mlp_normed, layer["mlp_gate_w"], None)
            self._debug_sync(f"layer {layer_idx} mlp_gate")
            Ops.linear(up, mlp_normed, layer["mlp_up_w"], None)
            self._debug_sync(f"layer {layer_idx} mlp_up")
            swiglu_out = Tensor((len(inputs), self.local_di), self.meta.dtype, self.device, self.device_id)
            Ops.swiglu(swiglu_out, gate, up)
            self._debug_sync(f"layer {layer_idx} swiglu")
            mlp_partial = Tensor((len(inputs), self.meta.hs), self.meta.dtype, self.device, self.device_id)
            Ops.linear(mlp_partial, swiglu_out, layer["mlp_down_w"], None)
            self._debug_sync(f"layer {layer_idx} mlp_down")
            self._all_reduce_tensor(mlp_partial)
            self._debug_sync(f"layer {layer_idx} mlp_all_reduce")
            Ops.add(hidden, hidden, mlp_partial)
            self._debug_sync(f"layer {layer_idx} mlp_residual")
            _tp_log(self.rank, f"layer {layer_idx} done")

        self._cur_pos += len(inputs)
        _tp_log(self.rank, f"generate_next layers done cur_pos={self._cur_pos}")

        next_token = -1
        if self.rank == 0:
            normed = Tensor(hidden.shape(), self.meta.dtype, self.device, self.device_id)
            Ops.rms_norm(normed, hidden, self.out_norm_w, self.meta.epsilon)
            last_hidden = normed.slice(0, len(inputs) - 1, len(inputs)).view(1, self.meta.hs)
            logits = Tensor((1, self.meta.voc), self.meta.dtype, self.device, self.device_id)
            Ops.linear(logits, last_hidden, self.out_embed, None)
            next_token = Ops.sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)

        return self._broadcast_token(next_token)

    def _debug_sync(self, label: str) -> None:
        if os.getenv("LLAISYS_TP_DEBUG", "").lower() not in {"1", "true", "yes", "on"}:
            return
        self.runtime.set_device(self.device_id)
        self.runtime.device_synchronize()
        _tp_log(self.rank, f"{label} ok")

    def _all_reduce_tensor(self, tensor: Tensor) -> None:
        buffer = self._to_torch(tensor)
        dist.all_reduce(buffer)
        self._from_torch(buffer, tensor)

    def _broadcast_token(self, token: int) -> int:
        value = torch.tensor([int(token) if self.rank == 0 else 0], dtype=torch.int64, device=self.cuda_device)
        dist.broadcast(value, src=0)
        return int(value.item())

    def _embedding(self, token_ids: List[int]) -> Tensor:
        tokens = self._make_int64_tensor(np.asarray(token_ids, dtype=np.int64))
        hidden = Tensor((len(token_ids), self.meta.hs), self.meta.dtype, self.device, self.device_id)
        Ops.embedding(hidden, tokens, self.in_embed)
        return hidden

    def _to_torch(self, tensor: Tensor) -> torch.Tensor:
        torch_tensor = torch.empty(tensor.shape(), dtype=self.torch_dtype, device=self.cuda_device)
        self.runtime.set_device(self.device_id)
        self.runtime.memcpy_sync(
            torch_tensor.data_ptr(),
            tensor.data_ptr(),
            torch_tensor.numel() * torch_tensor.element_size(),
            MemcpyKind.D2D,
        )
        return torch_tensor

    def _from_torch(self, torch_tensor: torch.Tensor, tensor: Tensor) -> None:
        self.runtime.set_device(self.device_id)
        self.runtime.memcpy_sync(
            tensor.data_ptr(),
            torch_tensor.data_ptr(),
            torch_tensor.numel() * torch_tensor.element_size(),
            MemcpyKind.D2D,
        )

    def _make_int64_tensor(self, values: np.ndarray) -> Tensor:
        tensor = Tensor(values.shape, DataType.I64, self.device, self.device_id)
        tensor.load(ctypes.c_void_p(values.ctypes.data))
        return tensor

    def _tensor_from_array(self, array: np.ndarray, *, dtype: Optional[DataType] = None) -> Tensor:
        target_dtype = self.meta.dtype if dtype is None else dtype
        tensor = Tensor(array.shape, target_dtype, self.device, self.device_id)
        tensor.load(ctypes.c_void_p(array.ctypes.data))
        self._tensors.append(tensor)
        return tensor

    def _load_weights(self) -> None:
        for file in sorted(self.model_path.glob("*.safetensors")):
            for key, array, source_dtype in self._iter_safetensors(file):
                self._assign_weight(key, array, source_dtype)

    def _assign_weight(self, name: str, array: np.ndarray, source_dtype: str) -> None:
        if name == "model.embed_tokens.weight":
            converted = self._convert_array(array, source_dtype)
            self.in_embed = self._tensor_from_array(np.ascontiguousarray(converted))
            return
        if name == "lm_head.weight":
            converted = self._convert_array(array, source_dtype)
            self.out_embed = self._tensor_from_array(np.ascontiguousarray(converted))
            return
        if name == "model.norm.weight":
            converted = self._convert_array(array, source_dtype)
            self.out_norm_w = self._tensor_from_array(np.ascontiguousarray(converted))
            return
        if not name.startswith("model.layers."):
            return

        parts = name.split(".")
        layer_idx = int(parts[2])
        suffix = ".".join(parts[3:])
        layer = self.layers[layer_idx]

        if suffix == "input_layernorm.weight":
            layer["attn_norm_w"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(array, source_dtype)))
            return
        if suffix == "post_attention_layernorm.weight":
            layer["mlp_norm_w"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(array, source_dtype)))
            return
        if suffix == "self_attn.q_proj.weight":
            shard = self._shard_rows(array)
            layer["attn_q_w"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(shard, source_dtype)))
            return
        if suffix == "self_attn.q_proj.bias":
            shard = self._shard_rows_bias(array)
            layer["attn_q_b"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(shard, source_dtype)))
            return
        if suffix == "self_attn.k_proj.weight":
            shard = self._shard_rows(array, kv=True)
            layer["attn_k_w"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(shard, source_dtype)))
            return
        if suffix == "self_attn.k_proj.bias":
            shard = self._shard_rows_bias(array, kv=True)
            layer["attn_k_b"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(shard, source_dtype)))
            return
        if suffix == "self_attn.v_proj.weight":
            shard = self._shard_rows(array, kv=True)
            layer["attn_v_w"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(shard, source_dtype)))
            return
        if suffix == "self_attn.v_proj.bias":
            shard = self._shard_rows_bias(array, kv=True)
            layer["attn_v_b"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(shard, source_dtype)))
            return
        if suffix == "self_attn.o_proj.weight":
            shard = self._shard_cols(array, self.local_q_dim)
            layer["attn_o_w"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(shard, source_dtype)))
            return
        if suffix == "mlp.gate_proj.weight":
            shard = self._shard_rows(array, mlp=True)
            layer["mlp_gate_w"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(shard, source_dtype)))
            return
        if suffix == "mlp.up_proj.weight":
            shard = self._shard_rows(array, mlp=True)
            layer["mlp_up_w"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(shard, source_dtype)))
            return
        if suffix == "mlp.down_proj.weight":
            shard = self._shard_cols(array, self.local_di)
            layer["mlp_down_w"] = self._tensor_from_array(np.ascontiguousarray(self._convert_array(shard, source_dtype)))
            return

    def _shard_rows(self, array: np.ndarray, *, kv: bool = False, mlp: bool = False) -> np.ndarray:
        if kv:
            chunk = self.local_kv_dim
        elif mlp:
            chunk = self.local_di
        else:
            chunk = self.local_q_dim
        start = self.rank * chunk
        end = start + chunk
        return np.asarray(array[start:end, :])

    def _shard_rows_bias(self, array: np.ndarray, *, kv: bool = False) -> np.ndarray:
        chunk = self.local_kv_dim if kv else self.local_q_dim
        start = self.rank * chunk
        end = start + chunk
        return np.asarray(array[start:end])

    def _shard_cols(self, array: np.ndarray, chunk: int) -> np.ndarray:
        start = self.rank * chunk
        end = start + chunk
        return np.asarray(array[:, start:end])

    def _iter_safetensors(self, path: Path):
        with open(path, "rb") as handle:
            length_bytes = handle.read(8)
            if not length_bytes:
                return
            header_size = struct.unpack("<Q", length_bytes)[0]
            header = json.loads(handle.read(header_size))
            fileno = handle.fileno()
            total_size = os.fstat(fileno).st_size
            mm = mmap.mmap(fileno, total_size, access=mmap.ACCESS_READ)
            data_start = 8 + header_size
            try:
                for key, info in header.items():
                    if key == "__metadata__":
                        continue
                    dtype_str = str(info["dtype"])
                    shape = tuple(info["shape"])
                    start, end = info["data_offsets"]
                    abs_start = data_start + start
                    if dtype_str in {"BF16", "bfloat16"}:
                        arr = np.array(
                            np.frombuffer(mm, dtype=np.uint16, count=(end - start) // 2, offset=abs_start),
                            copy=True,
                        ).reshape(shape)
                        yield key, arr, "bf16"
                    elif dtype_str in {"F32", "float32"}:
                        arr = np.array(
                            np.frombuffer(mm, dtype=np.float32, count=(end - start) // 4, offset=abs_start),
                            copy=True,
                        ).reshape(shape)
                        yield key, arr, "f32"
                    elif dtype_str in {"F16", "float16"}:
                        arr = np.array(
                            np.frombuffer(mm, dtype=np.float16, count=(end - start) // 2, offset=abs_start),
                            copy=True,
                        ).reshape(shape)
                        yield key, arr, "f16"
            finally:
                mm.close()

    def _convert_array(self, arr: np.ndarray, source_dtype: str) -> np.ndarray:
        if self.meta.dtype == DataType.F32:
            return Qwen2._to_float32_array(arr, source_dtype)
        if self.meta.dtype == DataType.F16:
            return Qwen2._to_float16_array(arr, source_dtype)
        if self.meta.dtype == DataType.BF16:
            return Qwen2._to_bfloat16_bytes(arr, source_dtype)
        raise ValueError(f"Unsupported model dtype: {self.meta.dtype}")


def _tensor_parallel_worker_main(
    rank: int,
    world_size: int,
    model_path: str,
    device_ids: List[int],
    master_addr: str,
    master_port: int,
    command_queue,
    result_queue,
) -> None:
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    torch.cuda.set_device(device_ids[rank])
    _tp_log(rank, f"worker start device={device_ids[rank]}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{device_ids[rank]}"),
    )
    worker = None
    try:
        worker = _ShardedQwen2Rank(model_path, rank, world_size, device_ids[rank])
        _tp_log(rank, "model loaded")
        dist.barrier()
        if rank == 0:
            result_queue.put({"status": "ready"})

        while True:
            message = command_queue.get()
            command = str(message["command"])
            seq = int(message["seq"])
            _tp_log(rank, f"command {command} seq={seq}")

            if command == "shutdown":
                if rank == 0:
                    result_queue.put({"seq": seq, "status": "ok"})
                break
            if command == "reset":
                worker.reset()
                if rank == 0:
                    result_queue.put({"seq": seq, "status": "ok"})
                continue
            if command == "truncate":
                worker.truncate(int(message["position"]))
                if rank == 0:
                    result_queue.put({"seq": seq, "status": "ok"})
                continue
            if command == "generate_next":
                if bool(message.get("reset_state", False)):
                    worker.reset()
                token = worker.generate_next(
                    message["inputs"],
                    top_k=int(message["top_k"]),
                    top_p=float(message["top_p"]),
                    temperature=float(message["temperature"]),
                )
                if rank == 0:
                    result_queue.put({"seq": seq, "status": "ok", "token": int(token)})
                continue
            raise ValueError(f"Unknown tensor-parallel command: {command}")
    except Exception as exc:  # pragma: no cover - defensive worker path
        _tp_log(rank, f"worker exception {repr(exc)}")
        result_queue.put({"status": "error", "rank": rank, "error": repr(exc)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TensorParallelQwen2:
    SUPPORTED_MODEL_TYPES: ClassVar[set[str]] = {"qwen2"}

    @classmethod
    def supports_model_type(cls, model_type: str) -> bool:
        return str(model_type).lower() in cls.SUPPORTED_MODEL_TYPES

    def __init__(
        self,
        model_path: str,
        device: DeviceType = DeviceType.NVIDIA,
        device_id: int = 0,
        *,
        tp_size: int = 2,
        tp_device_ids: Optional[Sequence[int]] = None,
    ):
        if device != DeviceType.NVIDIA:
            raise ValueError("TensorParallelQwen2 currently supports only NVIDIA devices")
        if int(tp_size) <= 1:
            raise ValueError("tp_size must be greater than 1 for tensor parallel inference")

        self.model_path = str(model_path)
        self.device = device
        self.device_id = int(device_id)
        self.device_ids = _resolve_tp_device_ids(device_id, tp_size, tp_device_ids)
        self.tp_size = len(self.device_ids)

        with open(Path(model_path) / "config.json", "r") as handle:
            config = json.load(handle)
        self.config = config
        self.model_type = str(config.get("model_type", "")).lower()
        if not self.supports_model_type(self.model_type):
            raise ValueError(f"Unsupported tensor-parallel model type: {self.model_type or '<missing model_type>'}")

        dtype = Qwen2._runtime_dtype(str(config.get("torch_dtype", "float32")), DeviceType.NVIDIA)
        eos_token = config.get("eos_token_id", 151643)
        if isinstance(eos_token, list):
            eos_token = eos_token[0] if eos_token else 151643
        self.meta = _TensorParallelMeta(
            dtype=dtype,
            nlayer=int(config.get("num_hidden_layers", 24)),
            hs=int(config.get("hidden_size", 2048)),
            nh=int(config.get("num_attention_heads", 16)),
            nkvh=int(config.get("num_key_value_heads", int(config.get("num_attention_heads", 16)))),
            dh=int(config.get("hidden_size", 2048)) // int(config.get("num_attention_heads", 16)),
            di=int(config.get("intermediate_size", 11008)),
            maxseq=int(config.get("max_position_embeddings", 8192)),
            voc=int(config.get("vocab_size", 151936)),
            epsilon=float(config.get("rms_norm_eps", 1e-6)),
            theta=float(config.get("rope_theta", 1000000.0)),
            end_token=int(eos_token),
        )
        _validate_tp_size(self.meta, self.tp_size)

        ctx = mp.get_context("spawn")
        self._command_queues = [ctx.Queue() for _ in range(self.tp_size)]
        self._result_queue = ctx.Queue()
        self._processes = []
        self._closed = False
        self._seq = 0
        self._master_addr = "127.0.0.1"
        self._master_port = _find_free_port()

        for rank in range(self.tp_size):
            process = ctx.Process(
                target=_tensor_parallel_worker_main,
                args=(
                    rank,
                    self.tp_size,
                    self.model_path,
                    self.device_ids,
                    self._master_addr,
                    self._master_port,
                    self._command_queues[rank],
                    self._result_queue,
                ),
                daemon=True,
            )
            process.start()
            self._processes.append(process)

        self._await_ready()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._request("shutdown")
        except Exception:
            pass
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
        self._processes.clear()

    def reset(self) -> None:
        self._request("reset")

    def truncate(self, position: int) -> None:
        self._request("truncate", position=int(position))

    def generate_next(
        self,
        inputs: Sequence[int],
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
        *,
        reset_state: bool = False,
    ) -> int:
        if not inputs:
            raise ValueError("inputs must not be empty")
        response = self._request(
            "generate_next",
            inputs=[int(token) for token in inputs],
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            reset_state=bool(reset_state),
        )
        return int(response["token"])

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 20,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        *,
        reset_state: bool = True,
    ) -> List[int]:
        if not inputs:
            raise ValueError("inputs must not be empty")
        if reset_state:
            self.reset()

        generated: List[int] = []
        token_source = list(inputs)
        for _ in range(int(max_new_tokens)):
            next_token = self.generate_next(
                token_source,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                reset_state=False,
            )
            generated.append(next_token)
            token_source = [next_token]
            if next_token == self.meta.end_token:
                break
        return list(inputs) + generated

    def stream_generate(
        self,
        inputs: Sequence[int],
        *,
        tokenizer=None,
        max_new_tokens: int = 20,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        reset_state: bool = True,
    ) -> Iterator[Tuple[int, str]]:
        if reset_state:
            self.reset()
        generated: List[int] = []
        previous_text = ""
        token_source = list(inputs)

        for _ in range(int(max_new_tokens)):
            next_token = self.generate_next(
                token_source,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                reset_state=False,
            )
            generated.append(next_token)
            token_source = [next_token]

            text_chunk = ""
            if tokenizer is not None:
                decoded = tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                text_chunk = decoded[len(previous_text):] if decoded.startswith(previous_text) else decoded
                previous_text = decoded

            yield next_token, text_chunk
            if next_token == self.meta.end_token:
                break

    def _request(self, command: str, **payload):
        self._seq += 1
        message = _normalize_command_payload(command, self._seq, **payload)
        for queue_ in self._command_queues:
            queue_.put(message)

        deadline_s = 600.0
        waited_s = 0.0
        while True:
            try:
                response = self._result_queue.get(timeout=5)
                break
            except queue.Empty:
                waited_s += 5.0
                dead = {
                    process.pid: process.exitcode
                    for process in self._processes
                    if process.exitcode not in (None, 0)
                }
                if dead:
                    raise RuntimeError(
                        f"Tensor-parallel worker exited while handling `{command}`: {dead}"
                    )
                if waited_s >= deadline_s:
                    raise TimeoutError(f"Timed out waiting for tensor-parallel command `{command}`")

        if response.get("status") == "error":
            raise RuntimeError(response.get("error", "Unknown tensor-parallel worker error"))
        if response.get("seq") != self._seq:
            raise RuntimeError(f"Unexpected tensor-parallel response ordering: {response}")
        return response

    def _await_ready(self) -> None:
        while True:
            try:
                ready = self._result_queue.get(timeout=5)
            except queue.Empty:
                dead = [process.pid for process in self._processes if process.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"Tensor-parallel worker exited before ready: pids={dead}")
                continue
            if ready.get("status") == "ready":
                return
            if ready.get("status") == "error":
                raise RuntimeError(
                    f"Tensor-parallel worker rank {ready.get('rank')} failed to start: {ready.get('error')}"
                )
            raise RuntimeError(f"Unexpected tensor-parallel startup response: {ready}")
