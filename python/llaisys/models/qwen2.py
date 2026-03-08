from typing import Sequence
import ctypes
import json
from pathlib import Path

import numpy as np
import safetensors
import torch

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor


def _np_dtype(dtype: DataType):
    if dtype == DataType.BF16:
        return np.dtype("bfloat16")
    if dtype == DataType.F16:
        return np.float16
    if dtype == DataType.F32:
        return np.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _llaisys_dtype(dtype_str: str) -> DataType:
    if dtype_str in ("bfloat16", "bf16"):
        return DataType.BF16
    if dtype_str in ("float16", "fp16", "f16"):
        return DataType.F16
    if dtype_str in ("float32", "fp32", "f32"):
        return DataType.F32
    return DataType.F32


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.config = config  # 保存 config 供后续使用

        hs = int(config["hidden_size"])
        nlayer = int(config["num_hidden_layers"])
        nh = int(config["num_attention_heads"])
        nkvh = int(config.get("num_key_value_heads", nh))
        di = int(config["intermediate_size"])
        maxseq = int(config.get("max_position_embeddings", 4096))
        voc = int(config["vocab_size"])
        epsilon = float(config.get("rms_norm_eps", 1e-5))
        theta = float(config.get("rope_theta", 10000.0))

        eos_token = config.get("eos_token_id", None)
        if isinstance(eos_token, list):
            end_token = int(eos_token[0])
        elif eos_token is None:
            end_token = -1
        else:
            end_token = int(eos_token)

        dtype = _llaisys_dtype(str(config.get("torch_dtype", "float32")))
        # 强制使用F32因为权重在加载时都转换为F32了
        dtype = DataType.F32
        try:
            np.dtype("bfloat16")
            self._np_bf16 = True
        except TypeError:
            self._np_bf16 = False
        dh = hs // nh

        meta = LlaisysQwen2Meta(
            dtype,
            nlayer,
            hs,
            nh,
            nkvh,
            dh,
            di,
            maxseq,
            voc,
            epsilon,
            theta,
            end_token,
        )

        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), ctypes.c_int(device), device_ids, ctypes.c_int(1)
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")
        self._end_token = end_token
        self._dtype = dtype
        self._nlayer = nlayer

        self._load_weights(model_path)

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _load_tensor(self, tensor_handle, array: np.ndarray):
        # 确保内存连续
        array = np.ascontiguousarray(array)
        # 直接调用 tensorLoad C API，避免创建临时 Tensor 对象（会触发析构器导致双重释放）
        LIB_LLAISYS.tensorLoad(tensor_handle, array.ctypes.data_as(ctypes.c_void_p))

    def _load_weights(self, model_path: Path):
        found_out_embed = False

        def get_weight(kind: int, layer: int = 0):
            return LIB_LLAISYS.llaisysQwen2ModelGetWeight(self._model, kind, layer)

        for file in sorted(model_path.glob("*.safetensors")):
            try:
                # Try numpy first (faster for most dtypes)
                data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            except Exception:
                # Fall back to torch for BF16 support
                data_ = safetensors.safe_open(file, framework="pt", device="cpu")
                
            for name_ in data_.keys():
                try:
                    arr = data_.get_tensor(name_)
                except Exception:
                    # If get_tensor fails with numpy framework, reload with torch
                    data_ = safetensors.safe_open(file, framework="pt", device="cpu")
                    arr = data_.get_tensor(name_)
                
                # Convert to numpy as F32
                if isinstance(arr, torch.Tensor):
                    # Convert torch tensor to numpy - always use float32 for consistency
                    arr = arr.float().cpu().numpy().astype(np.float32)
                else:
                    # numpy array
                    arr = arr.astype(np.float32)

                if name_ == "model.embed_tokens.weight":
                    self._load_tensor(get_weight(0), arr)
                    if not found_out_embed:
                        self._load_tensor(get_weight(1), arr)
                    continue

                if name_ == "lm_head.weight":
                    self._load_tensor(get_weight(1), arr)
                    found_out_embed = True
                    continue

                if name_ == "model.norm.weight":
                    self._load_tensor(get_weight(2), arr)
                    continue

                if not name_.startswith("model.layers."):
                    continue

                parts = name_.split(".")
                if len(parts) < 4:
                    continue
                layer_id = int(parts[2])
                rest = ".".join(parts[3:])

                if rest == "input_layernorm.weight":
                    self._load_tensor(get_weight(3, layer_id), arr)
                elif rest == "self_attn.q_proj.weight":
                    self._load_tensor(get_weight(4, layer_id), arr)
                elif rest == "self_attn.q_proj.bias":
                    self._load_tensor(get_weight(5, layer_id), arr)
                elif rest == "self_attn.k_proj.weight":
                    self._load_tensor(get_weight(6, layer_id), arr)
                elif rest == "self_attn.k_proj.bias":
                    self._load_tensor(get_weight(7, layer_id), arr)
                elif rest == "self_attn.v_proj.weight":
                    self._load_tensor(get_weight(8, layer_id), arr)
                elif rest == "self_attn.v_proj.bias":
                    self._load_tensor(get_weight(9, layer_id), arr)
                elif rest == "self_attn.o_proj.weight":
                    self._load_tensor(get_weight(10, layer_id), arr)
                elif rest == "post_attention_layernorm.weight":
                    self._load_tensor(get_weight(11, layer_id), arr)
                elif rest == "mlp.gate_proj.weight":
                    self._load_tensor(get_weight(12, layer_id), arr)
                elif rest == "mlp.up_proj.weight":
                    self._load_tensor(get_weight(13, layer_id), arr)
                elif rest == "mlp.down_proj.weight":
                    self._load_tensor(get_weight(14, layer_id), arr)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # 当前实现为 argmax 采样（top_k=1）
        tokens = list(inputs)
        max_new_tokens = 128 if max_new_tokens is None else int(max_new_tokens)

        top_k = int(top_k)
        top_p = float(top_p)
        temperature = float(temperature)

        for _ in range(max_new_tokens):
            arr = (ctypes.c_int64 * len(tokens))(*tokens)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model,
                    arr,
                    len(tokens),
                    ctypes.c_int(top_k),
                    ctypes.c_float(top_p),
                    ctypes.c_float(temperature),
                )
            )
            tokens.append(next_token)
            if self._end_token >= 0 and next_token == self._end_token:
                break

        return tokens
