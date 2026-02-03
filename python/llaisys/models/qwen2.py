from typing import Sequence, Dict
from pathlib import Path
import json
import ctypes

import torch
import safetensors

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.models import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor


_TORCH_DTYPE_TO_LLAISYS = {
    torch.float16: DataType.F16,
    torch.float32: DataType.F32,
    torch.bfloat16: DataType.BF16,
    torch.int64: DataType.I64,
}

_LLAISYS_DTYPE_TO_TORCH = {
    DataType.F16: torch.float16,
    DataType.F32: torch.float32,
    DataType.BF16: torch.bfloat16,
}


def _llaisys_dtype_from_torch(dtype: torch.dtype) -> DataType:
    if dtype not in _TORCH_DTYPE_TO_LLAISYS:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return _TORCH_DTYPE_TO_LLAISYS[dtype]


def _llaisys_dtype_from_config(dtype_name: str) -> DataType:
    name = (dtype_name or "").lower()
    if name in {"bfloat16", "bf16"}:
        return DataType.BF16
    if name in {"float16", "fp16", "f16"}:
        return DataType.F16
    if name in {"float32", "fp32", "f32"}:
        return DataType.F32
    raise ValueError(f"Unsupported config dtype: {dtype_name}")


def _torch_dtype_from_llaisys(dtype: DataType) -> torch.dtype:
    if dtype not in _LLAISYS_DTYPE_TO_TORCH:
        raise ValueError(f"Unsupported llaisys dtype: {dtype}")
    return _LLAISYS_DTYPE_TO_TORCH[dtype]


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"config.json not found under: {model_path}")

        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        self._device = device
        self._device_id = 0
        self._meta = self._build_meta(cfg)

        device_ids = (ctypes.c_int * 1)(self._device_id)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta),
            self._device,
            device_ids,
            ctypes.c_int(1),
        )

        self._weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._tensor_store: Dict[str, Tensor] = {}
        self._load_weights(model_path)

    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _build_meta(self, cfg: dict) -> LlaisysQwen2Meta:
        meta = LlaisysQwen2Meta()
        meta.dtype = _llaisys_dtype_from_config(cfg.get("torch_dtype", "bfloat16"))
        meta.nlayer = int(cfg["num_hidden_layers"])
        meta.hs = int(cfg["hidden_size"])
        meta.nh = int(cfg["num_attention_heads"])
        meta.nkvh = int(cfg.get("num_key_value_heads", meta.nh))
        meta.dh = int(meta.hs // meta.nh)
        meta.di = int(cfg["intermediate_size"])
        meta.maxseq = int(cfg.get("max_position_embeddings", cfg.get("max_seq_len", 2048)))
        meta.voc = int(cfg["vocab_size"])
        meta.epsilon = float(cfg.get("rms_norm_eps", cfg.get("layer_norm_eps", 1e-5)))
        meta.theta = float(cfg.get("rope_theta", 10000.0))

        eos = cfg.get("eos_token_id", -1)
        if isinstance(eos, list):
            eos = eos[0] if eos else -1
        meta.end_token = int(eos)
        return meta

    def _alloc_and_load_tensor(self, name: str, tensor: torch.Tensor) -> Tensor:
        target_torch_dtype = _torch_dtype_from_llaisys(DataType(self._meta.dtype))
        cpu_tensor = tensor.contiguous().to(dtype=target_torch_dtype, device="cpu")
        dtype = _llaisys_dtype_from_torch(cpu_tensor.dtype)
        shape = list(cpu_tensor.shape)
        llaisys_tensor = Tensor(
            shape=shape,
            dtype=dtype,
            device=self._device,
            device_id=self._device_id,
        )
        llaisys_tensor.load(ctypes.c_void_p(cpu_tensor.data_ptr()))
        self._tensor_store[name] = llaisys_tensor
        return llaisys_tensor

    def _assign_weight(self, name: str, llaisys_tensor: Tensor):
        weights = self._weights_ptr.contents

        if name == "model.embed_tokens.weight":
            weights.in_embed = llaisys_tensor.lib_tensor()
            return
        if name == "lm_head.weight":
            weights.out_embed = llaisys_tensor.lib_tensor()
            return
        if name == "model.norm.weight":
            weights.out_norm_w = llaisys_tensor.lib_tensor()
            return

        if not name.startswith("model.layers."):
            return

        parts = name.split(".")
        if len(parts) < 4:
            return
        try:
            layer = int(parts[2])
        except ValueError:
            return

        suffix = ".".join(parts[3:])
        if suffix == "input_layernorm.weight":
            weights.attn_norm_w[layer] = llaisys_tensor.lib_tensor()
            return
        if suffix == "self_attn.q_proj.weight":
            weights.attn_q_w[layer] = llaisys_tensor.lib_tensor()
            return
        if suffix == "self_attn.q_proj.bias":
            weights.attn_q_b[layer] = llaisys_tensor.lib_tensor()
            return
        if suffix == "self_attn.k_proj.weight":
            weights.attn_k_w[layer] = llaisys_tensor.lib_tensor()
            return
        if suffix == "self_attn.k_proj.bias":
            weights.attn_k_b[layer] = llaisys_tensor.lib_tensor()
            return
        if suffix == "self_attn.v_proj.weight":
            weights.attn_v_w[layer] = llaisys_tensor.lib_tensor()
            return
        if suffix == "self_attn.v_proj.bias":
            weights.attn_v_b[layer] = llaisys_tensor.lib_tensor()
            return
        if suffix == "self_attn.o_proj.weight":
            weights.attn_o_w[layer] = llaisys_tensor.lib_tensor()
            return
        if suffix == "post_attention_layernorm.weight":
            weights.mlp_norm_w[layer] = llaisys_tensor.lib_tensor()
            return
        if suffix == "mlp.gate_proj.weight":
            weights.mlp_gate_w[layer] = llaisys_tensor.lib_tensor()
            return
        if suffix == "mlp.up_proj.weight":
            weights.mlp_up_w[layer] = llaisys_tensor.lib_tensor()
            return
        if suffix == "mlp.down_proj.weight":
            weights.mlp_down_w[layer] = llaisys_tensor.lib_tensor()
            return

    def _load_weights(self, model_path: Path):
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                tensor = data_.get_tensor(name_)
                llaisys_tensor = self._alloc_and_load_tensor(name_, tensor)
                self._assign_weight(name_, llaisys_tensor)

        weights = self._weights_ptr.contents
        if not weights.out_embed and weights.in_embed:
            weights.out_embed = weights.in_embed

        self._check_weights()

    def _check_weights(self):
        weights = self._weights_ptr.contents
        missing = []

        def _need(cond, name):
            if not cond:
                missing.append(name)

        _need(weights.in_embed, "model.embed_tokens.weight")
        _need(weights.out_embed, "lm_head.weight (or tied)")
        _need(weights.out_norm_w, "model.norm.weight")

        for i in range(int(self._meta.nlayer)):
            _need(weights.attn_norm_w[i], f"layers.{i}.input_layernorm.weight")
            _need(weights.attn_q_w[i], f"layers.{i}.self_attn.q_proj.weight")
            _need(weights.attn_k_w[i], f"layers.{i}.self_attn.k_proj.weight")
            _need(weights.attn_v_w[i], f"layers.{i}.self_attn.v_proj.weight")
            _need(weights.attn_o_w[i], f"layers.{i}.self_attn.o_proj.weight")
            _need(weights.mlp_norm_w[i], f"layers.{i}.post_attention_layernorm.weight")
            _need(weights.mlp_gate_w[i], f"layers.{i}.mlp.gate_proj.weight")
            _need(weights.mlp_up_w[i], f"layers.{i}.mlp.up_proj.weight")
            _need(weights.mlp_down_w[i], f"layers.{i}.mlp.down_proj.weight")

        if missing:
            raise RuntimeError("Missing weights:\\n" + "\\n".join(missing))

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        tokens = list(inputs)
        for _ in range(max_new_tokens):
            arr = (ctypes.c_int64 * len(tokens))(*tokens)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, ctypes.c_size_t(len(tokens)))
            )
            tokens.append(next_token)
            if next_token == int(self._meta.end_token):
                break

        return tokens
