from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights

from pathlib import Path
from ctypes import c_int, c_int64, c_size_t, POINTER, byref
import json
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # Load config
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract model parameters
        hidden_size = config["hidden_size"]
        num_attention_heads = config["num_attention_heads"]
        num_key_value_heads = config["num_key_value_heads"]
        head_dim = hidden_size // num_attention_heads

        # Create meta structure
        self._meta = LlaisysQwen2Meta()
        self._meta.dtype = DataType.BF16.value
        self._meta.nlayer = config["num_hidden_layers"]
        self._meta.hs = hidden_size
        self._meta.nh = num_attention_heads
        self._meta.nkvh = num_key_value_heads
        self._meta.dh = head_dim
        self._meta.di = config["intermediate_size"]
        self._meta.maxseq = min(config.get("max_position_embeddings", 131072), 4096)
        self._meta.voc = config["vocab_size"]
        self._meta.epsilon = config.get("rms_norm_eps", 1e-6)
        self._meta.theta = config.get("rope_theta", 10000.0)
        self._meta.end_token = config.get("eos_token_id", 151643)

        self._device = device
        self._nlayer = self._meta.nlayer

        # Create model
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self._meta),
            device.value,
            device_ids,
            1
        )

        # Get weights pointer
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._weights = weights_ptr.contents

        # Load weights from safetensors
        self._load_weights(model_path)

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _load_weights(self, model_path: Path):
        """Load weights from safetensors files."""
        import torch

        # Collect all tensors from safetensors files
        all_tensors = {}
        for file in sorted(model_path.glob("*.safetensors")):
            data = safetensors.safe_open(file, framework="pt", device="cpu")
            for name in data.keys():
                all_tensors[name] = data.get_tensor(name)

        # Helper to load a tensor
        def load_tensor(tensor_handle, tensor_data):
            # Convert to contiguous bfloat16 and get raw bytes
            data = tensor_data.to(torch.bfloat16).contiguous()
            data_ptr = data.data_ptr()
            LIB_LLAISYS.tensorLoad(tensor_handle, data_ptr)

        # Load embedding weights
        load_tensor(self._weights.in_embed, all_tensors["model.embed_tokens.weight"])
        load_tensor(self._weights.out_embed, all_tensors["lm_head.weight"])
        load_tensor(self._weights.out_norm_w, all_tensors["model.norm.weight"])

        # Load per-layer weights
        for i in range(self._nlayer):
            prefix = f"model.layers.{i}."

            load_tensor(self._weights.attn_norm_w[i],
                       all_tensors[prefix + "input_layernorm.weight"])
            load_tensor(self._weights.attn_q_w[i],
                       all_tensors[prefix + "self_attn.q_proj.weight"])
            load_tensor(self._weights.attn_q_b[i],
                       all_tensors[prefix + "self_attn.q_proj.bias"])
            load_tensor(self._weights.attn_k_w[i],
                       all_tensors[prefix + "self_attn.k_proj.weight"])
            load_tensor(self._weights.attn_k_b[i],
                       all_tensors[prefix + "self_attn.k_proj.bias"])
            load_tensor(self._weights.attn_v_w[i],
                       all_tensors[prefix + "self_attn.v_proj.weight"])
            load_tensor(self._weights.attn_v_b[i],
                       all_tensors[prefix + "self_attn.v_proj.bias"])
            load_tensor(self._weights.attn_o_w[i],
                       all_tensors[prefix + "self_attn.o_proj.weight"])
            load_tensor(self._weights.mlp_norm_w[i],
                       all_tensors[prefix + "post_attention_layernorm.weight"])
            load_tensor(self._weights.mlp_gate_w[i],
                       all_tensors[prefix + "mlp.gate_proj.weight"])
            load_tensor(self._weights.mlp_up_w[i],
                       all_tensors[prefix + "mlp.up_proj.weight"])
            load_tensor(self._weights.mlp_down_w[i],
                       all_tensors[prefix + "mlp.down_proj.weight"])

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # Reset KV cache for new generation
        LIB_LLAISYS.llaisysQwen2ModelResetCache(self._model)

        # Convert input to ctypes array
        input_len = len(inputs)
        input_arr = (c_int64 * input_len)(*inputs)

        # Output tokens list (starts with input)
        output_tokens = list(inputs)

        # First forward pass with all input tokens
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model,
            input_arr,
            c_size_t(input_len)
        )
        output_tokens.append(next_token)

        # Generate remaining tokens one by one
        for _ in range(max_new_tokens - 1):
            if next_token == self._meta.end_token:
                break

            # Single token input
            single_token = (c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                single_token,
                c_size_t(1)
            )
            output_tokens.append(next_token)

        return output_tokens
