"""
Qwen2 模型的 Python 封装：通过 LLAISYS C++ 后端进行推理。

本模块不依赖 PyTorch 做推理，仅用 C 动态库（llaisys.dll / libllaisys.so）实现
前向计算。权重从 safetensors 文件加载；若为 bfloat16，需用 PyTorch 读取后转成
numpy 再灌入后端。
"""
from typing import Sequence
import ctypes
from ctypes import byref, cast, c_float, c_int, c_int64, c_size_t, c_ulonglong, POINTER

# 张量并行时传 device_ids

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights, LlaisysQwen2Model_t
from ..libllaisys.llaisys_types import DataType

from pathlib import Path
import json
import numpy as np
import safetensors

try:
    import torch
except ImportError:
    torch = None


def _weight_key_to_handle(weights_ptr, nlayer: int):
    """
    生成「safetensors 中的权重 key」到「C 侧权重句柄」的映射。

    权重文件名（key）与 Qwen2/HuggingFace 命名一致；部分 key 有别名（如 model.norm.w）
    以兼容 ModelScope 等不同来源的 checkpoint。

    Yields:
        (key, handle): key 为 safetensors 中的张量名，handle 为 C 侧 LlaisysQwen2Weights 中对应成员的指针。
    """
    w = weights_ptr.contents
    # 词嵌入与输出层
    yield "model.embed_tokens.weight", w.in_embed
    yield "model.norm.weight", w.out_norm_w
    yield "model.norm.w", w.out_norm_w  # 部分 checkpoint 用 model.norm.w
    yield "lm_head.weight", w.out_embed
    # 每一层的 attention / MLP 权重与 bias
    for i in range(nlayer):
        yield f"model.layers.{i}.input_layernorm.weight", w.attn_norm_w[i]
        yield f"model.layers.{i}.self_attn.q_proj.weight", w.attn_q_w[i]
        yield f"model.layers.{i}.self_attn.q_proj.bias", w.attn_q_b[i]
        yield f"model.layers.{i}.self_attn.k_proj.weight", w.attn_k_w[i]
        yield f"model.layers.{i}.self_attn.k_proj.bias", w.attn_k_b[i]
        yield f"model.layers.{i}.self_attn.v_proj.weight", w.attn_v_w[i]
        yield f"model.layers.{i}.self_attn.v_proj.bias", w.attn_v_b[i]
        yield f"model.layers.{i}.self_attn.o_proj.weight", w.attn_o_w[i]
        yield f"model.layers.{i}.post_attention_layernorm.weight", w.mlp_norm_w[i]
        yield f"model.layers.{i}.mlp.gate_proj.weight", w.mlp_gate_w[i]
        yield f"model.layers.{i}.mlp.up_proj.weight", w.mlp_up_w[i]
        yield f"model.layers.{i}.mlp.down_proj.weight", w.mlp_down_w[i]


def _shard_weight_for_tp(key: str, arr: np.ndarray, tp_rank: int, tp_world_size: int,
                         nh: int, nkvh: int, dh: int, di: int, hs: int, nlayer: int):
    """
    张量并行：按 key 对权重做行/列切分，返回当前 rank 应加载的 shard。
    列并行（输出维切分）：q/k/v_proj, gate/up_proj -> 切行。
    行并行（输入维切分）：o_proj, down_proj -> 切列。
    """
    if tp_world_size <= 1:
        return arr
    w = tp_world_size
    r = tp_rank
    arr = np.ascontiguousarray(arr)
    if "q_proj.weight" in key or "q_proj.bias" in key:
        # [nh*dh, hs] or [nh*dh]
        size = nh * dh
        step = size // w
        if "bias" in key:
            return arr[r * step : (r + 1) * step].copy()
        return arr[r * step : (r + 1) * step, :].copy()
    if "k_proj.weight" in key or "k_proj.bias" in key or "v_proj.weight" in key or "v_proj.bias" in key:
        size = nkvh * dh
        step = size // w
        if "bias" in key:
            return arr[r * step : (r + 1) * step].copy()
        return arr[r * step : (r + 1) * step, :].copy()
    if "o_proj.weight" in key:
        # [hs, nh*dh] 行并行
        size = nh * dh
        step = size // w
        return arr[:, r * step : (r + 1) * step].copy()
    if "gate_proj.weight" in key or "up_proj.weight" in key:
        step = di // w
        return arr[r * step : (r + 1) * step, :].copy()
    if "down_proj.weight" in key:
        step = di // w
        return arr[:, r * step : (r + 1) * step].copy()
    return arr


def _bf16_bytes_to_float32(raw: bytes) -> np.ndarray:
    """将 safetensors 中的 bf16 原始字节转为 float32 numpy（无 torch 依赖）。"""
    n = len(raw) // 2
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = (u16.astype(np.uint32) << 16)
    return np.frombuffer(u32.tobytes(), dtype=np.float32).copy()


def _read_safetensors_header(fpath: Path):
    """返回 (data_start_offset, key -> {dtype, shape, data_offsets})，便于无 torch 时读 bf16。"""
    with open(fpath, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header_json = f.read(header_len).decode("utf-8")
        data_start = 8 + (header_len + 7) // 8 * 8
    header = json.loads(header_json)
    key_to_meta = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        if isinstance(v, dict) and "dtype" in v and "shape" in v and "data_offsets" in v:
            key_to_meta[k] = v
    return data_start, key_to_meta


def _load_bf16_weights_from_safetensors_no_torch(
    fpath: Path,
    key_to_handle: dict,
    key_to_meta: dict,
    data_start: int,
    tp_world_size: int,
    tp_rank: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    dh: int,
    intermediate_size: int,
    hidden_size: int,
    num_hidden_layers: int,
) -> set:
    """从 safetensors 文件按 key 读 bf16 原始数据，转 float32 后灌入后端（不依赖 torch）。"""
    loaded = set()
    with open(fpath, "rb") as f:
        for key, handle in key_to_handle.items():
            if key not in key_to_meta:
                continue
            meta = key_to_meta[key]
            if meta.get("dtype") != "BF16":
                continue
            shape = meta["shape"]
            start, end = meta["data_offsets"]
            offset_in_file = data_start + start
            size_bytes = end - start
            f.seek(offset_in_file)
            raw = f.read(size_bytes)
            arr = _bf16_bytes_to_float32(raw).reshape(shape)
            arr = np.ascontiguousarray(arr)
            if tp_world_size > 1:
                arr = _shard_weight_for_tp(
                    key, arr, tp_rank, tp_world_size,
                    num_attention_heads, num_key_value_heads, dh,
                    intermediate_size, hidden_size, num_hidden_layers,
                )
            _numpy_to_backend(arr, handle)
            loaded.add(key)
    return loaded


def _numpy_to_backend(arr: np.ndarray, tensor_handle) -> None:
    """
    将 numpy 数组拷贝到 LLAISYS 后端张量（CPU 或设备内存）。

    若数组为 float32，会按 bfloat16 的「高 16 位」方式截断后传入后端，
    以兼容从 bfloat16 转成 float32 再传过来的权重。

    Args:
        arr: 主机侧 numpy 数组，需为连续内存。
        tensor_handle: C 侧张量句柄（LlaisysTensor*），由 tensorLoad 写入。
    """
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.float32:
        # float32 视为“从 bf16 转来的”，取高 16 位作为 bf16 比特表示
        arr_bf16 = (arr.view(np.uint32) >> 16).astype(np.uint16)
        LIB_LLAISYS.tensorLoad(tensor_handle, arr_bf16.ctypes.data)
    elif arr.dtype == np.uint16 or arr.dtype == np.float16:
        LIB_LLAISYS.tensorLoad(tensor_handle, arr.ctypes.data)
    else:
        LIB_LLAISYS.tensorLoad(tensor_handle, arr.ctypes.data)


class Qwen2:
    """
    Qwen2 模型的 Python 封装类。

    通过 LLAISYS C 接口创建模型、加载 safetensors 权重，并对外提供 generate()
    做自回归生成。推理全程在 C++ 后端执行，Python 只做配置、权重加载和循环调用 Infer。
    """

    def __init__(
        self,
        model_path,
        device: DeviceType = DeviceType.CPU,
        max_batch_size: int = 1,
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ):
        """
        从本地目录加载 Qwen2 模型：读 config、创建 C 模型、灌入权重。

        Args:
            model_path: 模型目录路径（需含 config.json 和 *.safetensors）。
            device: 运行设备，如 DeviceType.CPU 或 DeviceType.NVIDIA。
            max_batch_size: KV-Cache 槽位数，用于连续批处理；1 为单序列（默认）。
            tp_rank: 张量并行 rank（0..tp_world_size-1），默认 0。
            tp_world_size: 张量并行 world size，1 表示非分布式，默认 1。
        """
        model_path = Path(model_path)

        # ---------- 1. 读取 config.json ----------
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        hidden_size = config["hidden_size"]
        num_hidden_layers = config["num_hidden_layers"]
        num_attention_heads = config["num_attention_heads"]
        num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
        intermediate_size = config["intermediate_size"]
        vocab_size = config["vocab_size"]
        rms_norm_eps = float(config.get("rms_norm_eps", 1e-6))
        rope_theta = float(config.get("rope_theta", 10000.0))
        eos_id = config.get("eos_token_id", config.get("bos_token_id", 151643))
        max_position = config.get("max_position_embeddings", 131072)
        maxseq = min(4096, max_position)

        # 解析 dtype：支持 config 里 "dtype" 或 "torch_dtype"（如 "bfloat16" / "float16"）
        cfg_dtype = config.get("dtype", config.get("torch_dtype", "bfloat16"))
        if isinstance(cfg_dtype, str) and "bfloat" in cfg_dtype.lower():
            dtype = DataType.BF16
        elif isinstance(cfg_dtype, str) and "float16" in cfg_dtype.lower():
            dtype = DataType.F16
        else:
            dtype = DataType.BF16

        # 每个 attention 头的维度
        dh = hidden_size // num_attention_heads

        # ---------- 2. 组装 C 侧元信息并创建模型 ----------
        # max_batch_size：连续批处理时 KV 槽位数，默认 1 保持单序列行为
        # tp_rank / tp_world_size：张量并行（项目#5），默认 0/1 表示单卡
        meta = LlaisysQwen2Meta(
            dtype=dtype,
            nlayer=num_hidden_layers,
            hs=hidden_size,
            nh=num_attention_heads,
            nkvh=num_key_value_heads,
            dh=dh,
            di=intermediate_size,
            maxseq=maxseq,
            voc=vocab_size,
            max_batch_size=max_batch_size,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            epsilon=rms_norm_eps,
            theta=rope_theta,
            end_token=eos_id,
        )

        # TP 时每进程通常通过 CUDA_VISIBLE_DEVICES 只暴露一张卡，故传 device_ids=None 用默认 0 即可
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            device,
            None,
            0,
        )
        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate failed")

        self._end_token = eos_id
        self._nlayer = num_hidden_layers
        self._max_batch_size = max_batch_size

        # ---------- 3. 从 safetensors 加载权重到 C 侧 ----------
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        key_to_handle = dict(_weight_key_to_handle(weights_ptr, num_hidden_layers))
        loaded_keys = set()

        # bfloat16 时可用 PyTorch 读；若无 torch（如 spawn 子进程避免 nccl 冲突）则走纯 Python 解析 safetensors
        use_pt = torch is not None
        use_bf16_no_torch = (dtype == DataType.BF16 and torch is None)

        safetensor_files = sorted(model_path.glob("*.safetensors"))
        for idx, fpath in enumerate(safetensor_files):
            if torch is not None and idx == 0:
                pass  # 首次用 torch 打开文件时可能触发 c10 等，便于定位崩溃
            print(f"  Loading weights: {fpath.name} ({idx + 1}/{len(safetensor_files)})", flush=True)
            if use_bf16_no_torch:
                data_start, key_to_meta = _read_safetensors_header(fpath)
                loaded = _load_bf16_weights_from_safetensors_no_torch(
                    fpath, key_to_handle, key_to_meta, data_start,
                    tp_world_size, tp_rank,
                    num_attention_heads, num_key_value_heads, dh,
                    intermediate_size, hidden_size, num_hidden_layers,
                )
                loaded_keys.update(loaded)
                continue
            with safetensors.safe_open(
                fpath, framework="pt" if use_pt else "numpy", device="cpu"
            ) as data:
                for key in data.keys():
                    if key not in key_to_handle:
                        continue
                    handle = key_to_handle[key]
                    t = data.get_tensor(key)
                    if use_pt:
                        if t.dtype == torch.bfloat16:
                            arr = t.float().numpy()
                        else:
                            arr = t.numpy()
                    else:
                        arr = np.ascontiguousarray(t)
                    arr = np.ascontiguousarray(arr)
                    if tp_world_size > 1:
                        arr = _shard_weight_for_tp(
                            key, arr, tp_rank, tp_world_size,
                            num_attention_heads, num_key_value_heads, dh,
                            intermediate_size, hidden_size, num_hidden_layers,
                        )
                    _numpy_to_backend(arr, handle)
                    loaded_keys.add(key)

        # 至少需要嵌入层权重，否则说明 key 对不上
        embed_loaded = "model.embed_tokens.weight" in loaded_keys
        if not embed_loaded:
            sample_keys = []
            for fpath in sorted(model_path.glob("*.safetensors")):
                with safetensors.safe_open(
                    fpath, framework="pt" if use_pt else "numpy", device="cpu"
                ) as data:
                    sample_keys.extend(list(data.keys())[:40])
                break
            raise RuntimeError(
                "No embedding weights loaded. Loaded %d keys; sample keys from file: %s"
                % (len(loaded_keys), sample_keys[:30])
            )

        # GPU 时：默认只缓存输出层（推理主体在 GPU，快）；若环境变量 LLAISYS_GPU_FULL_CPU=1 则全量缓存，整次前向在 CPU（慢但可规避其它 GPU 算子问题）
        if device == DeviceType.NVIDIA:
            import os
            if os.environ.get("LLAISYS_GPU_FULL_CPU") == "1":
                LIB_LLAISYS.llaisysQwen2ModelCacheAllWeightsOnCPU(self._model)
                print("  [Qwen2] GPU 已缓存全量权重到 CPU，推理全程在 CPU 上执行（LLAISYS_GPU_FULL_CPU=1）。", flush=True)
            else:
                LIB_LLAISYS.llaisysQwen2ModelCacheOutputLayerOnCPU(self._model)
                print("  [Qwen2] GPU：embedding 与输出层在 CPU，其余层在 GPU。", flush=True)

    def kv_cache_bytes(self, prefix_len: int) -> int:
        """存储前缀长度为 prefix_len 的 KV cache 所需字节数。"""
        return LIB_LLAISYS.llaisysQwen2ModelGetKVCacheBytes(self._model, prefix_len)

    def export_kv_cache(self) -> bytes:
        """导出当前 KV cache 到字节串（当前 cache_len 由 C 侧维护）。"""
        n = LIB_LLAISYS.llaisysQwen2ModelGetCacheLen(self._model)
        if n == 0:
            return b""
        size = LIB_LLAISYS.llaisysQwen2ModelGetKVCacheBytes(self._model, n)
        buf = (ctypes.c_byte * size)()
        LIB_LLAISYS.llaisysQwen2ModelExportKVCache(self._model, ctypes.cast(buf, ctypes.c_void_p))
        return bytes(buf)

    def reset_kv_cache(self) -> None:
        """将 KV cache 长度置 0，新请求全量 prefill 前调用，避免沿用上一轮状态。"""
        LIB_LLAISYS.llaisysQwen2ModelResetKVCache(self._model)

    def import_kv_cache(self, data: bytes, prefix_len: int) -> None:
        """从字节串导入前缀长度为 prefix_len 的 KV cache；之后可对 suffix 做 prefill。"""
        if prefix_len == 0 or not data:
            return
        expected = self.kv_cache_bytes(prefix_len)
        if len(data) < expected:
            raise ValueError(f"import_kv_cache: need {expected} bytes, got {len(data)}")
        buf = (ctypes.c_byte * len(data))()
        ctypes.memmove(ctypes.addressof(buf), data, len(data))
        LIB_LLAISYS.llaisysQwen2ModelImportKVCache(
            self._model, ctypes.cast(buf, ctypes.c_void_p), prefix_len
        )

    def export_kv_cache_slot(self, slot_id: int) -> bytes:
        """导出指定 slot 的 KV cache 到字节串（用于连续批处理 + KV 池）。"""
        n = LIB_LLAISYS.llaisysQwen2ModelGetCacheLenSlot(self._model, slot_id)
        if n == 0:
            return b""
        size = LIB_LLAISYS.llaisysQwen2ModelGetKVCacheBytes(self._model, n)
        buf = (ctypes.c_byte * size)()
        LIB_LLAISYS.llaisysQwen2ModelExportKVCacheSlot(
            self._model, slot_id, ctypes.cast(buf, ctypes.c_void_p)
        )
        return bytes(buf)

    def import_kv_cache_slot(self, slot_id: int, data: bytes, prefix_len: int) -> None:
        """将字节串导入到指定 slot 的前缀 KV cache；之后可对该 slot 做 suffix prefill。"""
        if prefix_len == 0 or not data:
            return
        expected = self.kv_cache_bytes(prefix_len)
        if len(data) < expected:
            raise ValueError(
                f"import_kv_cache_slot: need {expected} bytes, got {len(data)}"
            )
        buf = (ctypes.c_byte * len(data))()
        ctypes.memmove(ctypes.addressof(buf), data, len(data))
        LIB_LLAISYS.llaisysQwen2ModelImportKVCacheSlot(
            self._model,
            slot_id,
            ctypes.cast(buf, ctypes.c_void_p),
            prefix_len,
        )

    @property
    def cache_len(self) -> int:
        """当前已写入 KV cache 的长度。"""
        return LIB_LLAISYS.llaisysQwen2ModelGetCacheLen(self._model)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: int = 0,
        prefix_len: int = 0,
    ):
        """
        自回归生成：从当前 token 序列出发，每次调用 C 的 Infer 得到下一个 token，直到 EOS 或达到 max_new_tokens。
        支持随机采样：temperature、top_k、top_p 会传入 C 侧；top_k=1 且 temperature 接近 0 时为 argmax 贪心。
        prefix_len>0 时表示已通过 import_kv_cache 导入前缀，仅对 inputs[prefix_len:] 做 suffix prefill 再 decode。

        Args:
            inputs: 初始 token id 序列（如 prompt 经 tokenizer 编码后的列表）。
            max_new_tokens: 最多新生成多少个 token。
            top_k: 保留概率最高的 k 个 token，<=0 表示不限制。
            top_p: nucleus 采样阈值，<=0 或 >=1 表示不限制。
            temperature: 温度，<=0 或极小为贪心。
            seed: 随机种子，0 表示每次随机。
            prefix_len: 若已 import_kv_cache，则为前缀长度；0 表示全量 prefill。

        Returns:
            完整 token 序列（inputs + 新生成的 token），包含 EOS 在内。
        """
        # 检查输入是否为空，如果连问题都没有，AI 没法往下接话
        if not inputs:
            raise ValueError("generate() called with empty input token list")

        import os
        if os.environ.get("LLAISYS_DEBUG"):
            print(f"[LLAISYS] Qwen2.generate() n_inputs={len(inputs)} prefix_len={prefix_len} max_new_tokens={max_new_tokens}")

        tokens = list(inputs)
        if prefix_len == 0:
            self.reset_kv_cache()
        # 首步：全量 prefill 或 suffix prefill（需先 import_kv_cache）
        if prefix_len > 0:
            if prefix_len >= len(tokens):
                raise ValueError("prefix_len must be < len(inputs)")
            suffix = tokens[prefix_len:]
            n = len(suffix)
            token_arr = (c_int64 * n)(*suffix)
            next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                cast(token_arr, POINTER(c_int64)),
                n,
                c_float(temperature),
                c_int(top_k),
                c_float(top_p),
                c_ulonglong(seed),
            )
            if next_tok == -1:
                raise RuntimeError("llaisysQwen2ModelInfer failed (returned -1)")
            tokens.append(next_tok)
            if next_tok == self._end_token:
                return tokens
            n_decoded = 1
        else:
            n = len(tokens)
            token_arr = (c_int64 * n)(*tokens)
            next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                cast(token_arr, POINTER(c_int64)),
                n,
                c_float(temperature),
                c_int(top_k),
                c_float(top_p),
                c_ulonglong(seed),
            )
            if next_tok == -1:
                raise RuntimeError("llaisysQwen2ModelInfer failed (returned -1)")
            tokens.append(next_tok)
            if next_tok == self._end_token:
                return tokens
            n_decoded = 1

        # 后续为 decode 步：每次只传最后一个 token（ntoken=1）
        for _ in range(max_new_tokens - n_decoded):
            token_arr = (c_int64 * 1)(tokens[-1])
            next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                cast(token_arr, POINTER(c_int64)),
                1,
                c_float(temperature),
                c_int(top_k),
                c_float(top_p),
                c_ulonglong(seed),
            )
            if next_tok == -1:
                raise RuntimeError("llaisysQwen2ModelInfer failed (returned -1)")
            tokens.append(next_tok)
            if next_tok == self._end_token:
                break
        return tokens

    @property
    def end_token(self) -> int:
        """EOS token id，用于流式生成时判断结束。"""
        return self._end_token

    @property
    def max_batch_size(self) -> int:
        """KV-Cache 槽位数，供连续批处理 Engine 使用。"""
        return getattr(self, "_max_batch_size", 1)

    def next_token(
        self,
        token_ids: Sequence[int],
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.8,
        seed: int = 0,
    ) -> int:
        """
        单步推理：给定当前 token 序列，返回下一个 token id。
        供流式输出或外部自回归循环使用。
        """
        if not token_ids:
            raise ValueError("next_token() requires non-empty token_ids")
        n = len(token_ids)
        token_arr = (c_int64 * n)(*token_ids)
        next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model,
            cast(token_arr, POINTER(c_int64)),
            n,
            c_float(temperature),
            c_int(top_k),
            c_float(top_p),
            c_ulonglong(seed),
        )
        if next_tok == -1:
            raise RuntimeError("llaisysQwen2ModelInfer failed (returned -1)")
        return next_tok

    def infer_hybrid(
        self,
        token_ids: Sequence[int],
        temperature: float = 0.0,
        top_k: int = 1,
        top_p: float = 1.0,
        seed: int = 0,
        gpu_up_to_layer: int = -1,
    ) -> int:
        """
        诊断用：前 (gpu_up_to_layer+1) 层在 GPU 上跑，其余在 CPU。需已调用 CacheAllWeightsOnCPU。
        gpu_up_to_layer=-1：全 CPU；=0：仅 embedding 在 GPU；=1：embedding+layer0 在 GPU；依此类推。
        调用前会 reset_kv_cache。
        """
        self.reset_kv_cache()
        n = len(token_ids)
        token_arr = (c_int64 * n)(*token_ids)
        next_tok = LIB_LLAISYS.llaisysQwen2ModelInferHybrid(
            self._model,
            cast(token_arr, POINTER(c_int64)),
            n,
            c_float(temperature),
            c_int(top_k),
            c_float(top_p),
            c_ulonglong(seed),
            c_int(gpu_up_to_layer),
        )
        if next_tok == -1:
            raise RuntimeError("llaisysQwen2ModelInferHybrid failed (returned -1)")
        return next_tok

    # ===== Python 的魔法方法：析构函数 =====
    def __del__(self):
        """析构时释放 C 侧模型，避免泄漏。""" #
        # 当 Python 里的 Qwen2 对象不再被使用，准备被垃圾回收时，会自动触发这个函数
        # 它负责打电话通知 C++ 侧：“我要下线了，你把那些占了几个 G 内存的模型张量（Tensor）全删了吧！”
        if getattr(self, "_model", None) is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model) # 调用 C++ 的销毁接口
            self._model = None # 清空指针，防止重复释放报错
    