from typing import Sequence, Iterator
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights

from pathlib import Path
import ctypes
import json
import safetensors
import torch


class Qwen2:

    DTYPE_MAP = {
        "bfloat16": DataType.BF16,
        "float16": DataType.F16,
        "float32": DataType.F32,
    }

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        with open(model_path / "config.json") as f:
            config = json.load(f)

        torch_dtype = config.get("torch_dtype", "bfloat16")
        dtype = self.DTYPE_MAP.get(torch_dtype, DataType.BF16)

        nh = config["num_attention_heads"]
        nkvh = config["num_key_value_heads"]
        hs = config["hidden_size"]
        dh = hs // nh

        meta = LlaisysQwen2Meta()
        meta.dtype = dtype
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = hs
        meta.nh = nh
        meta.nkvh = nkvh
        meta.dh = dh
        meta.di = config["intermediate_size"]
        meta.maxseq = min(config.get("max_position_embeddings", 131072), 4096)
        meta.voc = config["vocab_size"]
        meta.epsilon = config.get("rms_norm_eps", 1e-6)
        meta.theta = config.get("rope_theta", 10000.0)
        meta.end_token = config.get("eos_token_id", 151643)
        if isinstance(meta.end_token, list):
            meta.end_token = meta.end_token[0]

        self._nlayer = meta.nlayer
        self._end_token = meta.end_token
        self._device = device

        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            ctypes.c_int(device),
            device_ids,
            ctypes.c_int(1),
        )

        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        weights = weights_ptr.contents

        name_map = self._build_name_map(weights)

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                if name_ in name_map:
                    tensor_handle = name_map[name_]
                    t = data_.get_tensor(name_).contiguous()
                    LIB_LLAISYS.tensorLoad(tensor_handle, ctypes.c_void_p(t.data_ptr()))

    def _build_name_map(self, weights: LlaisysQwen2Weights):
        m = {}
        m["model.embed_tokens.weight"] = weights.in_embed
        m["lm_head.weight"] = weights.out_embed
        m["model.norm.weight"] = weights.out_norm_w

        for i in range(self._nlayer):
            prefix = f"model.layers.{i}"
            m[f"{prefix}.input_layernorm.weight"] = weights.attn_norm_w[i]
            m[f"{prefix}.self_attn.q_proj.weight"] = weights.attn_q_w[i]
            m[f"{prefix}.self_attn.q_proj.bias"] = weights.attn_q_b[i]
            m[f"{prefix}.self_attn.k_proj.weight"] = weights.attn_k_w[i]
            m[f"{prefix}.self_attn.k_proj.bias"] = weights.attn_k_b[i]
            m[f"{prefix}.self_attn.v_proj.weight"] = weights.attn_v_w[i]
            m[f"{prefix}.self_attn.v_proj.bias"] = weights.attn_v_b[i]
            m[f"{prefix}.self_attn.o_proj.weight"] = weights.attn_o_w[i]
            m[f"{prefix}.post_attention_layernorm.weight"] = weights.mlp_norm_w[i]
            m[f"{prefix}.mlp.gate_proj.weight"] = weights.mlp_gate_w[i]
            m[f"{prefix}.mlp.up_proj.weight"] = weights.mlp_up_w[i]
            m[f"{prefix}.mlp.down_proj.weight"] = weights.mlp_down_w[i]

        return m

    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def reset_kvcache(self):
        LIB_LLAISYS.llaisysQwen2ModelResetKVCache(self._model)

    def _infer_one(self, token_ids, use_sample, temperature, top_k, top_p):
        arr = (ctypes.c_int64 * len(token_ids))(*token_ids)
        n = ctypes.c_size_t(len(token_ids))
        if use_sample:
            return LIB_LLAISYS.llaisysQwen2ModelInferSample(
                self._model, arr, n,
                ctypes.c_float(temperature),
                ctypes.c_int(top_k),
                ctypes.c_float(top_p),
            )
        else:
            return LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, n)

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

        use_sample = not (top_k == 1 and temperature == 1.0)
        tokens = list(inputs)

        next_token = self._infer_one(tokens, use_sample, temperature, top_k, top_p)
        tokens.append(next_token)

        for _ in range(max_new_tokens - 1):
            if next_token == self._end_token:
                break
            next_token = self._infer_one([next_token], use_sample, temperature, top_k, top_p)
            tokens.append(next_token)

        return tokens

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 512,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.8,
    ) -> Iterator[int]:
        use_sample = not (top_k == 1 and temperature == 1.0)

        next_token = self._infer_one(list(inputs), use_sample, temperature, top_k, top_p)
        yield next_token

        for _ in range(max_new_tokens - 1):
            if next_token == self._end_token:
                return
            next_token = self._infer_one([next_token], use_sample, temperature, top_k, top_p)
            yield next_token
