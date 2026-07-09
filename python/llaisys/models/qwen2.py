from typing import Sequence
from pathlib import Path
import json
import ctypes
import numpy as np
import safetensors

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import llaisysDeviceType_t, llaisysDataType_t
from ..libllaisys.models import LlaisysQwen2Meta, LlaisysQwen2Weights


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        print(f"Loading Qwen2 model from {model_path} on device {device.name}")

        self._device = device
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found under {model_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        torch_dtype = str(config.get("torch_dtype", "bfloat16")).lower()
        if "bfloat" in torch_dtype:
            dtype = DataType.BF16
        elif "float16" in torch_dtype or "fp16" in torch_dtype:
            dtype = DataType.F16
        else:
            dtype = DataType.F32

        nlayer = int(config["num_hidden_layers"])
        hs = int(config["hidden_size"])
        nh = int(config["num_attention_heads"])
        nkvh = int(config.get("num_key_value_heads", nh))
        dh = int(hs // nh)
        di = int(config["intermediate_size"])
        maxseq = int(config.get("max_position_embeddings", config.get("max_seq_len", 2048)))
        voc = int(config["vocab_size"])
        epsilon = float(config.get("rms_norm_eps", 1e-6))
        theta = float(config.get("rope_theta", 10000.0))

        eos_token = config.get("eos_token_id", None)
        if isinstance(eos_token, list):
            end_token = int(eos_token[0]) if eos_token else -1
        elif eos_token is None:
            end_token = -1
        else:
            end_token = int(eos_token)

        meta = LlaisysQwen2Meta(
            llaisysDataType_t(dtype),
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
            ctypes.byref(meta),
            llaisysDeviceType_t(device),
            device_ids,
            ctypes.c_int(1),
        )
        self._weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._weights: LlaisysQwen2Weights = self._weights_ptr.contents
        self._dtype = dtype
        self._end_token = end_token

        loaded = set()
        use_torch_fallback = False
        if self._dtype == DataType.BF16:
            try:
                np.dtype("bfloat16")
            except TypeError:
                use_torch_fallback = True

        for file in sorted(model_path.glob("*.safetensors")):
            if use_torch_fallback:
                import torch

                data_ = safetensors.safe_open(file, framework="pt", device="cpu")
                for name_ in data_.keys():
                    tensor = self._resolve_weight(name_)
                    if tensor is None:
                        continue
                    t = data_.get_tensor(name_)
                    if t.dtype != torch.bfloat16:
                        t = t.to(torch.bfloat16)
                    raw = t.view(torch.uint16).contiguous().cpu().numpy()
                    LIB_LLAISYS.tensorLoad(tensor, ctypes.c_void_p(raw.ctypes.data))
                    loaded.add(name_)
            else:
                data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
                for name_ in data_.keys():
                    tensor = self._resolve_weight(name_)
                    if tensor is None:
                        continue
                    arr = data_.get_tensor(name_)
                    self._load_tensor(tensor, arr)
                    loaded.add(name_)

        if "lm_head.weight" not in loaded:
            self._weights_ptr.contents.out_embed = self._weights_ptr.contents.in_embed

    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _np_dtype(self):
        if self._dtype == DataType.BF16:
            return np.dtype("bfloat16")
        if self._dtype == DataType.F16:
            return np.float16
        if self._dtype == DataType.F32:
            return np.float32
        raise ValueError(f"Unsupported dtype: {self._dtype}")

    def _load_tensor(self, tensor, arr):
        target_dtype = self._np_dtype()
        if arr.dtype != target_dtype:
            arr = arr.astype(target_dtype)
        arr = np.ascontiguousarray(arr)
        LIB_LLAISYS.tensorLoad(tensor, ctypes.c_void_p(arr.ctypes.data))

    def _resolve_weight(self, name: str):
        if name == "model.embed_tokens.weight":
            return self._weights.in_embed
        if name == "lm_head.weight":
            return self._weights.out_embed
        if name == "model.norm.weight":
            return self._weights.out_norm_w

        if name.startswith("model.layers."):
            parts = name.split(".")
            if len(parts) < 5:
                return None
            try:
                layer = int(parts[2])
            except ValueError:
                return None

            block = parts[3]
            if block == "input_layernorm" and parts[-1] == "weight":
                return self._weights.attn_norm_w[layer]
            if block in ("self_attn", "self_attention"):
                if len(parts) < 6:
                    return None
                proj = parts[4]
                param = parts[5]
                if proj == "q_proj":
                    return self._weights.attn_q_w[layer] if param == "weight" else self._weights.attn_q_b[layer]
                if proj == "k_proj":
                    return self._weights.attn_k_w[layer] if param == "weight" else self._weights.attn_k_b[layer]
                if proj == "v_proj":
                    return self._weights.attn_v_w[layer] if param == "weight" else self._weights.attn_v_b[layer]
                if proj == "o_proj" and param == "weight":
                    return self._weights.attn_o_w[layer]
            if block == "post_attention_layernorm" and parts[-1] == "weight":
                return self._weights.mlp_norm_w[layer]
            if block == "mlp":
                if len(parts) < 6:
                    return None
                proj = parts[4]
                param = parts[5]
                if proj == "gate_proj" and param == "weight":
                    return self._weights.mlp_gate_w[layer]
                if proj == "up_proj" and param == "weight":
                    return self._weights.mlp_up_w[layer]
                if proj == "down_proj" and param == "weight":
                    return self._weights.mlp_down_w[layer]

        return None

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
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                arr,
                ctypes.c_size_t(len(tokens)),
                ctypes.c_float(temperature),
                ctypes.c_int(top_k),
                ctypes.c_float(top_p),
            )
            tokens.append(int(next_token))
            if self._end_token >= 0 and next_token == self._end_token:
                break

        return tokens
