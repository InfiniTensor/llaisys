from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights

from pathlib import Path
import json
import ctypes
import safetensors.torch
import torch


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, device_id: int = 0):
        model_path = Path(model_path)

        # Read config.json
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract model parameters
        nlayer = config["num_hidden_layers"]
        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        nkvh = config["num_key_value_heads"]
        dh = hs // nh
        di = config["intermediate_size"]
        maxseq = config["max_position_embeddings"]
        voc = config["vocab_size"]
        epsilon = config["rms_norm_eps"]
        theta = config.get("rope_theta", 10000.0)
        end_token = config["eos_token_id"]

        # Create meta structure
        self._meta = LlaisysQwen2Meta(
            nlayer=nlayer,
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=dh,
            di=di,
            maxseq=maxseq,
            voc=voc,
            epsilon=epsilon,
            theta=theta,
            end_token=end_token,
        )

        self._device = device
        self._device_id = device_id
        self._nlayer = nlayer
        self._end_token = end_token

        # Create model
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta),
            device,
            device_id,
        )

        # Get weights pointer
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)

        # Load weights from safetensors
        self._load_weights(model_path)

    def _load_weights(self, model_path: Path):
        weights = self._weights.contents

        # Weight name mapping
        weight_map = {
            "model.embed_tokens.weight": weights.in_embed,
            "lm_head.weight": weights.out_embed,
            "model.norm.weight": weights.out_norm_w,
        }

        # Per-layer weight mapping
        for i in range(self._nlayer):
            weight_map[f"model.layers.{i}.input_layernorm.weight"] = weights.attn_norm_w[i]
            weight_map[f"model.layers.{i}.self_attn.q_proj.weight"] = weights.attn_q_w[i]
            weight_map[f"model.layers.{i}.self_attn.q_proj.bias"] = weights.attn_q_b[i]
            weight_map[f"model.layers.{i}.self_attn.k_proj.weight"] = weights.attn_k_w[i]
            weight_map[f"model.layers.{i}.self_attn.k_proj.bias"] = weights.attn_k_b[i]
            weight_map[f"model.layers.{i}.self_attn.v_proj.weight"] = weights.attn_v_w[i]
            weight_map[f"model.layers.{i}.self_attn.v_proj.bias"] = weights.attn_v_b[i]
            weight_map[f"model.layers.{i}.self_attn.o_proj.weight"] = weights.attn_o_w[i]
            weight_map[f"model.layers.{i}.post_attention_layernorm.weight"] = weights.mlp_norm_w[i]
            weight_map[f"model.layers.{i}.mlp.gate_proj.weight"] = weights.mlp_gate_w[i]
            weight_map[f"model.layers.{i}.mlp.up_proj.weight"] = weights.mlp_up_w[i]
            weight_map[f"model.layers.{i}.mlp.down_proj.weight"] = weights.mlp_down_w[i]

        # Load from safetensors files
        for file in sorted(model_path.glob("*.safetensors")):
            data = safetensors.torch.load_file(file, device="cpu")
            for name, tensor_data in data.items():
                if name in weight_map:
                    tensor_handle = weight_map[name]
                    # Ensure tensor is contiguous and get raw data pointer
                    tensor_data = tensor_data.contiguous()
                    # Load data into tensor
                    LIB_LLAISYS.tensorLoad(
                        tensor_handle,
                        ctypes.c_void_p(tensor_data.data_ptr()),
                    )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # Reset KV cache for new generation
        LIB_LLAISYS.llaisysQwen2ModelResetCache(self._model)

        # Convert inputs to ctypes array
        input_list = list(inputs)
        outputs = input_list.copy()

        # First inference with all input tokens
        input_array = (ctypes.c_int64 * len(input_list))(*input_list)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model,
            input_array,
            len(input_list),
        )
        outputs.append(next_token)

        # Continue generating
        tokens_generated = 1
        while max_new_tokens is None or tokens_generated < max_new_tokens:
            if next_token == self._end_token:
                break

            # Single token inference
            single_token = (ctypes.c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                single_token,
                1,
            )
            outputs.append(next_token)
            tokens_generated += 1

        return outputs

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
