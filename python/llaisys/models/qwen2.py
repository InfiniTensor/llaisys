from typing import Sequence
from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    Qwen2Meta,
    Qwen2Weights,
    llaisysQwen2Model_t,
)
from ..tensor import Tensor
from pathlib import Path
import safetensors
from ctypes import c_int64, c_int, c_void_p
import ctypes
import numpy as np


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # Load config.json to get model parameters
        import json
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create model metadata
        meta = Qwen2Meta()
        meta.dtype = DataType.F32.value  # Use float32 for now (BF16 converted to F32)
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = config["hidden_size"]
        meta.nh = config["num_attention_heads"]
        meta.nkvh = config.get("num_key_value_heads", meta.nh)
        meta.dh = meta.hs // meta.nh
        meta.di = config["intermediate_size"]
        meta.maxseq = config.get("max_position_embeddings", 32768)
        meta.voc = config["vocab_size"]
        meta.epsilon = config["rms_norm_eps"]
        meta.theta = config.get("rope_theta", 10000.0)
        meta.end_token = config.get("eos_token_id", 151645)
        
        # Create model
        device_id = c_int(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            meta,
            device.value,
            device_id,
            1
        )
        
        # Get weight tensors
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._meta = meta
        
        # Map weight names to tensors
        weight_map = {}
        
        # Load weights from safetensors files
        for file in sorted(model_path.glob("*.safetensors")):
            data = safetensors.safe_open(file, framework="pt", device="cpu")  # Use PyTorch framework
            for name in data.keys():
                weight_map[name] = data.get_tensor(name)
        
        # Load weights into model
        self._load_weight(weight_map, "model.embed_tokens.weight", self._weights.contents.in_embed)
        self._load_weight(weight_map, "lm_head.weight", self._weights.contents.out_embed)
        self._load_weight(weight_map, "model.norm.weight", self._weights.contents.out_norm_w)
        
        for i in range(meta.nlayer):
            self._load_weight(weight_map, f"model.layers.{i}.input_layernorm.weight", 
                            self._weights.contents.attn_norm_w[i])
            self._load_weight(weight_map, f"model.layers.{i}.self_attn.q_proj.weight", 
                            self._weights.contents.attn_q_w[i])
            self._load_weight(weight_map, f"model.layers.{i}.self_attn.q_proj.bias", 
                            self._weights.contents.attn_q_b[i])
            self._load_weight(weight_map, f"model.layers.{i}.self_attn.k_proj.weight", 
                            self._weights.contents.attn_k_w[i])
            self._load_weight(weight_map, f"model.layers.{i}.self_attn.k_proj.bias", 
                            self._weights.contents.attn_k_b[i])
            self._load_weight(weight_map, f"model.layers.{i}.self_attn.v_proj.weight", 
                            self._weights.contents.attn_v_w[i])
            self._load_weight(weight_map, f"model.layers.{i}.self_attn.v_proj.bias", 
                            self._weights.contents.attn_v_b[i])
            self._load_weight(weight_map, f"model.layers.{i}.self_attn.o_proj.weight", 
                            self._weights.contents.attn_o_w[i])
            self._load_weight(weight_map, f"model.layers.{i}.post_attention_layernorm.weight", 
                            self._weights.contents.mlp_norm_w[i])
            self._load_weight(weight_map, f"model.layers.{i}.mlp.gate_proj.weight", 
                            self._weights.contents.mlp_gate_w[i])
            self._load_weight(weight_map, f"model.layers.{i}.mlp.up_proj.weight", 
                            self._weights.contents.mlp_up_w[i])
            self._load_weight(weight_map, f"model.layers.{i}.mlp.down_proj.weight", 
                            self._weights.contents.mlp_down_w[i])
    
    def _load_weight(self, weight_map, name, tensor_handle):
        """Load weight data into a tensor"""
        if name not in weight_map:
            raise ValueError(f"Weight {name} not found in model")
        
        data = weight_map[name]
        
        # Convert PyTorch tensor to numpy if needed
        if hasattr(data, 'numpy'):
            # Handle BFloat16 - convert to float32 first then to numpy
            if hasattr(data, 'dtype') and 'bfloat16' in str(data.dtype):
                import torch
                data = data.to(torch.float32).numpy()
            else:
                data = data.numpy()
        
        # Convert to contiguous array if needed
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        # Create Tensor wrapper and load data
        tensor = Tensor(tensor=tensor_handle)
        tensor.load(data.ctypes.data_as(c_void_p))

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # For now, only support greedy decoding (argmax)
        if top_k != 1 or temperature != 1.0:
            print("Warning: Only argmax sampling is currently supported")
        
        output_tokens = list(inputs)
        
        # First forward pass with all input tokens
        if len(inputs) > 0:
            input_array = (c_int64 * len(inputs))(*inputs)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, input_array, len(inputs))
            output_tokens.append(next_token)
        
        # Generate new tokens one by one
        max_tokens = max_new_tokens if max_new_tokens else 128
        for _ in range(max_tokens - 1):
            # Forward with single token
            input_array = (c_int64 * 1)(output_tokens[-1])
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, input_array, 1)
            output_tokens.append(next_token)
            
            # Check for end token
            if next_token == self._meta.end_token:
                break
        
        return output_tokens

    def __del__(self):
        if hasattr(self, '_model') and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)

