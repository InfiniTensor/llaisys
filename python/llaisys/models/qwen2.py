from typing import Sequence
import warnings
from ctypes import byref, c_int, c_size_t, c_float, c_int64, c_uint32, c_void_p
import json
from pathlib import Path

import numpy as np
import safetensors

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    llaisysDeviceType_t,
    llaisysDataType_t,
    LlaisysQwen2Meta,
    LlaisysSamplingParams,
)


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # 实例化模型元信息
        config_path = model_path / "config.json"
        # 如果config.json不存在，则递归查找
        if not config_path.exists():
            candidates = list(model_path.rglob("config.json"))
            if not candidates:
                raise FileNotFoundError("config.json not found under model_path")
            config_path = candidates[0]
        # 读取配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # 解析数据类型
        torch_dtype = str(cfg.get("torch_dtype", "bfloat16")).lower()
        if "float32" in torch_dtype or torch_dtype in {"fp32", "f32"}:
            dtype = DataType.F32
        elif "float16" in torch_dtype or torch_dtype in {"fp16", "f16"}:
            dtype = DataType.F16
        else:
            dtype = DataType.BF16
        # 统一用 torch 读取 bfloat16，并降级为 float16，避免 numpy bfloat16 兼容问题
        use_torch_loader = False
        if dtype == DataType.BF16:
            dtype = DataType.F16
            use_torch_loader = True
        # 解析模型参数
        nlayer = int(cfg.get("num_hidden_layers", 0))
        hs = int(cfg.get("hidden_size", 0))
        nh = int(cfg.get("num_attention_heads", 0))
        nkvh = int(cfg.get("num_key_value_heads", nh))
        di = int(cfg.get("intermediate_size", 0))
        maxseq = int(cfg.get("max_position_embeddings", 0))
        voc = int(cfg.get("vocab_size", 0))
        epsilon = float(cfg.get("rms_norm_eps", 1e-6))
        theta = float(cfg.get("rope_theta", 10000.0))
        eos = cfg.get("eos_token_id", -1)
        # 解析结束token
        if isinstance(eos, list):
            end_token = int(eos[0]) if eos else -1
        else:
            end_token = int(eos)
        # 解析head_dim
        dh = int(cfg.get("head_dim", hs // nh if nh else 0))
        # 创建模型元信息结构体
        model_meta = LlaisysQwen2Meta(
            llaisysDataType_t(dtype),
            c_size_t(nlayer),
            c_size_t(hs),
            c_size_t(nh),
            c_size_t(nkvh),
            c_size_t(dh),
            c_size_t(di),
            c_size_t(maxseq),
            c_size_t(voc),
            c_float(epsilon),
            c_float(theta),
            c_int64(end_token),
        )
        # 创建模型实例
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(model_meta),
            llaisysDeviceType_t(device),
            device_ids,
            1,
        )
        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate failed")
        self._model_weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._meta = model_meta

        # 默认开启 KV-cache
        LIB_LLAISYS.llaisysQwen2ModelSetKVCacheEnabled(self._model, c_int(1))
        #
        def _dtype_to_llaisys(dtype: np.dtype) -> DataType:
            name = getattr(dtype, "name", str(dtype)).lower()
            if name in {"float32", "f4"}:
                return DataType.F32
            if name in {"float16", "f2"}:
                return DataType.F16
            if name in {"bfloat16", "bf16"}:
                return DataType.BF16
            if name in {"int64", "i8"}:
                return DataType.I64
            if name in {"int32", "i4"}:
                return DataType.I32
            if name in {"int16", "i2"}:
                return DataType.I16
            if name in {"int8", "i1"}:
                return DataType.I8
            if name in {"uint8", "u1"}:
                return DataType.U8
            raise ValueError(f"Unsupported dtype: {dtype}")

        def _create_tensor_from_numpy(arr: np.ndarray):
            arr = np.ascontiguousarray(arr)
            _shape = (c_size_t * arr.ndim)(*arr.shape)
            _dtype = _dtype_to_llaisys(arr.dtype)
            tensor = LIB_LLAISYS.tensorCreate(
                _shape,
                c_size_t(arr.ndim),
                llaisysDataType_t(_dtype),
                llaisysDeviceType_t(device),
                c_int(0),
            )
            LIB_LLAISYS.tensorLoad(tensor, c_void_p(arr.ctypes.data))
            return tensor

        # 加载模型权重
        for file in sorted(model_path.glob("*.safetensors")):
            if use_torch_loader:
                import torch
                data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            else:
                data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                ## TODO: load the model weights
                try:
                    arr = data_.get_tensor(name_)
                except TypeError:
                    # numpy 无法处理 bfloat16 时，回退到 torch
                    import torch
                    data_ = safetensors.safe_open(file, framework="pt", device="cpu")
                    arr = data_.get_tensor(name_)
                    use_torch_loader = True
                if use_torch_loader:
                    if arr.dtype == torch.bfloat16:
                        arr = arr.to(torch.float16)
                    arr = arr.cpu().numpy()
                tensor = _create_tensor_from_numpy(arr)
                w = self._model_weights.contents

                if name_ in {"model.embed_tokens.weight", "transformer.wte.weight"}:
                    w.in_embed = tensor
                    continue
                if name_ in {"lm_head.weight", "model.lm_head.weight"}:
                    w.out_embed = tensor
                    continue
                if name_ in {"model.norm.weight", "transformer.ln_f.weight"}:
                    w.out_norm_w = tensor
                    continue

                if name_.startswith("model.layers."):
                    parts = name_.split(".")
                    if len(parts) < 4:
                        continue
                    layer = int(parts[2])
                    sub = ".".join(parts[3:])

                    if sub == "input_layernorm.weight":
                        w.attn_norm_w[layer] = tensor
                    elif sub == "self_attn.q_proj.weight":
                        w.attn_q_w[layer] = tensor
                    elif sub == "self_attn.q_proj.bias":
                        w.attn_q_b[layer] = tensor
                    elif sub == "self_attn.k_proj.weight":
                        w.attn_k_w[layer] = tensor
                    elif sub == "self_attn.k_proj.bias":
                        w.attn_k_b[layer] = tensor
                    elif sub == "self_attn.v_proj.weight":
                        w.attn_v_w[layer] = tensor
                    elif sub == "self_attn.v_proj.bias":
                        w.attn_v_b[layer] = tensor
                    elif sub == "self_attn.o_proj.weight":
                        w.attn_o_w[layer] = tensor
                    elif sub == "post_attention_layernorm.weight":
                        w.mlp_norm_w[layer] = tensor
                    elif sub == "mlp.gate_proj.weight":
                        w.mlp_gate_w[layer] = tensor
                    elif sub == "mlp.up_proj.weight":
                        w.mlp_up_w[layer] = tensor
                    elif sub == "mlp.down_proj.weight":
                        w.mlp_down_w[layer] = tensor

        w = self._model_weights.contents
        if not w.out_embed and w.in_embed:
            w.out_embed = w.in_embed

    
    def generate(
        self,
        # 输入数组
        inputs: Sequence[int],
        # 最大token数
        max_new_tokens: int = None,
        # top-k 采样，1 表示贪心
        top_k: int = 1,
        # top-p 核采样阈值
        top_p: float = 0.8,
        # 温度系数，越小越保守
        temperature: float = 0.8,
        # 随机种子，0 表示随机
        seed: int = 0,
    ):
        tokens = list(inputs)
        if max_new_tokens is None:
            max_new_tokens = 128

        # prefill with full prompt
        token_buf = (c_int64 * len(tokens))(*tokens)
        next_token = int(
            LIB_LLAISYS.llaisysQwen2ModelPrefill(
                self._model,
                token_buf,
                c_size_t(len(tokens)),
            )
        )
        if next_token < 0:
            return tokens
        tokens.append(next_token)
        if self._meta.end_token >= 0 and next_token == self._meta.end_token:
            return tokens

        remaining = max_new_tokens - 1
        if remaining <= 0:
            return tokens

        # step with newly generated tokens only
        for _ in range(remaining):
            if next_token < 0:
                break
            if self._meta.end_token >= 0 and next_token == self._meta.end_token:
                break
            token_buf = (c_int64 * 1)(next_token)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelStep(
                    self._model,
                    token_buf,
                    c_size_t(1),
                )
            )
            if next_token < 0:
                break
            tokens.append(next_token)

        return tokens

    def prefill(self, inputs: Sequence[int]) -> int:
        tokens = list(inputs)
        token_buf = (c_int64 * len(tokens))(*tokens)
        return int(
            LIB_LLAISYS.llaisysQwen2ModelPrefill(
                self._model,
                token_buf,
                c_size_t(len(tokens)),
            )
        )

    def step(self, new_tokens: Sequence[int]) -> int:
        tokens = list(new_tokens)
        token_buf = (c_int64 * len(tokens))(*tokens)
        return int(
            LIB_LLAISYS.llaisysQwen2ModelStep(
                self._model,
                token_buf,
                c_size_t(len(tokens)),
            )
        )

    def infer(self, inputs: Sequence[int]) -> int:
        warnings.warn(
            "Qwen2.infer is deprecated; use prefill()/step() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.prefill(inputs)

    def reset_kv_cache(self):
        LIB_LLAISYS.llaisysQwen2ModelResetKVCache(self._model)
