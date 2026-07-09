from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor

from pathlib import Path
import safetensors
import json
import ctypes


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # Load config
        config_path = model_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create model meta
        meta = LlaisysQwen2Meta()
        meta.dtype = 19  # LLAISYS_DTYPE_BF16
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = config["hidden_size"]
        meta.nh = config["num_attention_heads"]
        meta.nkvh = config["num_key_value_heads"]
        meta.dh = config["hidden_size"] // config["num_attention_heads"]
        meta.di = config["intermediate_size"]
        meta.maxseq = config.get("max_position_embeddings", 32768)
        meta.voc = config["vocab_size"]
        meta.epsilon = config["rms_norm_eps"]
        meta.theta = config.get("rope_theta", 10000.0)
        meta.end_token = config.get("eos_token_id", 151643)

        # Create model
        device_id = 0
        device_ids = (ctypes.c_int * 1)(device_id)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), device.value, device_ids, 1
        )

        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        # Get weights structure
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        if not weights_ptr:
            raise RuntimeError("Failed to get model weights")

        self._weights = weights_ptr.contents
        self._meta = meta
        self._device = device

        # Load weights from safetensors
        self._load_weights(model_path)

    def _load_weights(self, model_path):
        """Load weights from safetensors files"""
        import torch
        import numpy as np
        import ctypes
        self.weight_tensors = []  # Keep references to prevent garbage collection
        weight_map = {}

        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    # Load as contiguous PyTorch tensor
                    pt_tensor = f.get_tensor(name).contiguous()
                    self.weight_tensors.append(pt_tensor)  # Keep alive
                    weight_map[name] = pt_tensor

        # Load embedding
        if "model.embed_tokens.weight" in weight_map:
            embed_data = weight_map["model.embed_tokens.weight"]
            tensor = Tensor.from_ptr(self._weights.in_embed)
            tensor.load(ctypes.c_void_p(embed_data.data_ptr()))

        # Load output norm and lm_head
        if "model.norm.weight" in weight_map:
            norm_data = weight_map["model.norm.weight"]
            tensor = Tensor.from_ptr(self._weights.out_norm_w)
            tensor.load(ctypes.c_void_p(norm_data.data_ptr()))

        if "lm_head.weight" in weight_map:
            lm_head_data = weight_map["lm_head.weight"]
            tensor = Tensor.from_ptr(self._weights.out_embed)
            tensor.load(ctypes.c_void_p(lm_head_data.data_ptr()))

        # Load per-layer weights
        for layer_idx in range(self._meta.nlayer):
            prefix = f"model.layers.{layer_idx}"

            # Attention norm
            if f"{prefix}.input_layernorm.weight" in weight_map:
                data = weight_map[f"{prefix}.input_layernorm.weight"]
                tensor = Tensor.from_ptr(self._weights.attn_norm_w[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

            # Q, K, V projections
            if f"{prefix}.self_attn.q_proj.weight" in weight_map:
                data = weight_map[f"{prefix}.self_attn.q_proj.weight"]
                tensor = Tensor.from_ptr(self._weights.attn_q_w[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

            if f"{prefix}.self_attn.q_proj.bias" in weight_map:
                data = weight_map[f"{prefix}.self_attn.q_proj.bias"]
                tensor = Tensor.from_ptr(self._weights.attn_q_b[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

            if f"{prefix}.self_attn.k_proj.weight" in weight_map:
                data = weight_map[f"{prefix}.self_attn.k_proj.weight"]
                tensor = Tensor.from_ptr(self._weights.attn_k_w[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

            if f"{prefix}.self_attn.k_proj.bias" in weight_map:
                data = weight_map[f"{prefix}.self_attn.k_proj.bias"]
                tensor = Tensor.from_ptr(self._weights.attn_k_b[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

            if f"{prefix}.self_attn.v_proj.weight" in weight_map:
                data = weight_map[f"{prefix}.self_attn.v_proj.weight"]
                tensor = Tensor.from_ptr(self._weights.attn_v_w[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

            if f"{prefix}.self_attn.v_proj.bias" in weight_map:
                data = weight_map[f"{prefix}.self_attn.v_proj.bias"]
                tensor = Tensor.from_ptr(self._weights.attn_v_b[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

            # O projection
            if f"{prefix}.self_attn.o_proj.weight" in weight_map:
                data = weight_map[f"{prefix}.self_attn.o_proj.weight"]
                tensor = Tensor.from_ptr(self._weights.attn_o_w[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

            # MLP norm
            if f"{prefix}.post_attention_layernorm.weight" in weight_map:
                data = weight_map[f"{prefix}.post_attention_layernorm.weight"]
                tensor = Tensor.from_ptr(self._weights.mlp_norm_w[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

            # MLP projections
            if f"{prefix}.mlp.gate_proj.weight" in weight_map:
                data = weight_map[f"{prefix}.mlp.gate_proj.weight"]
                tensor = Tensor.from_ptr(self._weights.mlp_gate_w[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

            if f"{prefix}.mlp.up_proj.weight" in weight_map:
                data = weight_map[f"{prefix}.mlp.up_proj.weight"]
                tensor = Tensor.from_ptr(self._weights.mlp_up_w[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

            if f"{prefix}.mlp.down_proj.weight" in weight_map:
                data = weight_map[f"{prefix}.mlp.down_proj.weight"]
                tensor = Tensor.from_ptr(self._weights.mlp_down_w[layer_idx])
                tensor.load(ctypes.c_void_p(data.data_ptr()))

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """Generate tokens using the model"""
        # For now, only support greedy decoding (top_k=1)
        if top_k != 1:
            raise NotImplementedError("Only greedy decoding (top_k=1) is supported")

        generated = list(inputs)
        max_gen = max_new_tokens if max_new_tokens else 100

        # First forward pass with full prompt
        input_array = (ctypes.c_int64 * len(generated))(*generated)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model, input_array, len(generated)
        )

        if next_token < 0 or next_token == self._meta.end_token:
            generated.append(next_token)
            return generated

        generated.append(next_token)

        # Subsequent passes: only send new token (using KV cache)
        for _ in range(max_gen - 1):
            # Only send the last generated token
            input_array = (ctypes.c_int64 * 1)(generated[-1])

            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, input_array, 1
            )

            if next_token < 0:
                break

            generated.append(next_token)

            # Check for EOS
            if next_token == self._meta.end_token:
                break

        return generated

    def __del__(self):
        if hasattr(self, '_model') and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
