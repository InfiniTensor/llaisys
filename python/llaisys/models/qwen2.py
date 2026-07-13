from typing import Sequence, List
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights

from pathlib import Path
import safetensors
import json
import ctypes
import re


class Qwen2:

    def __init__(
        self,
        model_path,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        model_path = Path(model_path)
        
        # Load config
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Extract config values
        self.num_layers = config["num_hidden_layers"]  # 28
        self.hidden_size = config["hidden_size"]       # 1536
        self.num_heads = config["num_attention_heads"] # 12
        self.num_kv_heads = config["num_key_value_heads"]  # 2
        self.head_dim = self.hidden_size // self.num_heads  # 128
        self.intermediate_size = config["intermediate_size"]  # 8960
        self.max_seq_len = config["max_position_embeddings"]  # 131072
        self.vocab_size = config["vocab_size"]  # 151936
        self.rms_norm_eps = config["rms_norm_eps"]  # 1e-6
        self.rope_theta = config["rope_theta"]  # 10000
        self.eos_token_id = config["eos_token_id"]  # 151643
        
        # Map torch dtype to llaisys dtype
        torch_dtype = config.get("torch_dtype", "float32")
        if torch_dtype == "bfloat16":
            self.dtype = DataType.BF16
        elif torch_dtype == "float16":
            self.dtype = DataType.F16
        else:
            self.dtype = DataType.F32
        
        self.device = device
        self.device_id = device_id
        self.rank = rank
        self.world_size = world_size
        
        # Create model meta
        meta = LlaisysQwen2Meta()
        meta.dtype = self.dtype
        meta.nlayer = self.num_layers
        meta.hs = self.hidden_size
        meta.nh = self.num_heads
        meta.nkvh = self.num_kv_heads
        meta.dh = self.head_dim
        meta.di = self.intermediate_size
        meta.maxseq = self.max_seq_len
        meta.voc = self.vocab_size
        meta.epsilon = self.rms_norm_eps
        meta.theta = self.rope_theta
        meta.end_token = self.eos_token_id
        
        # Create model
        device_ids = (ctypes.c_int * 1)(device_id)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            device,
            device_ids,
            1
        )
        
        # Get weights pointer
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        
        # Load weights from safetensors
        self._load_weights(model_path)
    
    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
    
    def _load_weights(self, model_path: Path):
        """Load weights from safetensors files."""
        for file in sorted(model_path.glob("*.safetensors")):
            data = safetensors.safe_open(file, framework="pt", device="cpu")
            for name in data.keys():
                tensor = data.get_tensor(name)
                self._load_weight(name, tensor)
    
    def _load_weight(self, name: str, tensor):
        """Load a single weight tensor."""
        import torch
        
        # Convert to contiguous and get raw data pointer
        tensor = self._slice_weight(name, tensor).contiguous()
        data_ptr = tensor.data_ptr()
        
        # Map weight name to model weight
        if name == "model.embed_tokens.weight":
            LIB_LLAISYS.tensorLoad(self._weights.in_embed, ctypes.c_void_p(data_ptr))
        elif name == "lm_head.weight":
            LIB_LLAISYS.tensorLoad(self._weights.out_embed, ctypes.c_void_p(data_ptr))
        elif name == "model.norm.weight":
            LIB_LLAISYS.tensorLoad(self._weights.out_norm_w, ctypes.c_void_p(data_ptr))
        else:
            # Parse layer index from name like "model.layers.0.xxx"
            match = re.match(r"model\.layers\.(\d+)\.(.*)", name)
            if match:
                layer_idx = int(match.group(1))
                weight_name = match.group(2)
                
                if weight_name == "input_layernorm.weight":
                    LIB_LLAISYS.tensorLoad(self._weights.attn_norm_w[layer_idx], ctypes.c_void_p(data_ptr))
                elif weight_name == "self_attn.q_proj.weight":
                    LIB_LLAISYS.tensorLoad(self._weights.attn_q_w[layer_idx], ctypes.c_void_p(data_ptr))
                elif weight_name == "self_attn.q_proj.bias":
                    LIB_LLAISYS.tensorLoad(self._weights.attn_q_b[layer_idx], ctypes.c_void_p(data_ptr))
                elif weight_name == "self_attn.k_proj.weight":
                    LIB_LLAISYS.tensorLoad(self._weights.attn_k_w[layer_idx], ctypes.c_void_p(data_ptr))
                elif weight_name == "self_attn.k_proj.bias":
                    LIB_LLAISYS.tensorLoad(self._weights.attn_k_b[layer_idx], ctypes.c_void_p(data_ptr))
                elif weight_name == "self_attn.v_proj.weight":
                    LIB_LLAISYS.tensorLoad(self._weights.attn_v_w[layer_idx], ctypes.c_void_p(data_ptr))
                elif weight_name == "self_attn.v_proj.bias":
                    LIB_LLAISYS.tensorLoad(self._weights.attn_v_b[layer_idx], ctypes.c_void_p(data_ptr))
                elif weight_name == "self_attn.o_proj.weight":
                    LIB_LLAISYS.tensorLoad(self._weights.attn_o_w[layer_idx], ctypes.c_void_p(data_ptr))
                elif weight_name == "post_attention_layernorm.weight":
                    LIB_LLAISYS.tensorLoad(self._weights.mlp_norm_w[layer_idx], ctypes.c_void_p(data_ptr))
                elif weight_name == "mlp.gate_proj.weight":
                    LIB_LLAISYS.tensorLoad(self._weights.mlp_gate_w[layer_idx], ctypes.c_void_p(data_ptr))
                elif weight_name == "mlp.up_proj.weight":
                    LIB_LLAISYS.tensorLoad(self._weights.mlp_up_w[layer_idx], ctypes.c_void_p(data_ptr))
                elif weight_name == "mlp.down_proj.weight":
                    LIB_LLAISYS.tensorLoad(self._weights.mlp_down_w[layer_idx], ctypes.c_void_p(data_ptr))

    def _slice_weight(self, name: str, tensor):
        if self.world_size == 1:
            return tensor

        def shard_dim(t, dim):
            assert t.shape[dim] % self.world_size == 0
            shard = t.shape[dim] // self.world_size
            start = self.rank * shard
            end = start + shard
            if dim == 0:
                return t[start:end]
            if dim == 1:
                return t[:, start:end]
            raise ValueError(f"Unsupported shard dim: {dim}")

        if name in {"model.embed_tokens.weight", "lm_head.weight"}:
            return shard_dim(tensor, 0)
        if name in {"model.norm.weight"}:
            return tensor

        match = re.match(r"model\.layers\.(\d+)\.(.*)", name)
        if not match:
            return tensor

        weight_name = match.group(2)
        if weight_name in {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        }:
            return tensor
        if weight_name in {
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
        }:
            return shard_dim(tensor, 0)
        if weight_name in {
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        }:
            assert tensor.shape[0] % self.world_size == 0
            shard = tensor.shape[0] // self.world_size
            start = self.rank * shard
            end = start + shard
            return tensor[start:end]
        if weight_name in {
            "self_attn.o_proj.weight",
            "mlp.down_proj.weight",
        }:
            return shard_dim(tensor, 1)
        return tensor

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> List[int]:
        """Generate tokens using argmax sampling (for testing)."""
        
        if max_new_tokens is None:
            max_new_tokens = 128
        
        # Convert inputs to list
        tokens = list(inputs)
        
        # First inference with all input tokens
        input_array = (ctypes.c_int64 * len(tokens))(*tokens)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model,
            input_array,
            len(tokens)
        )
        tokens.append(next_token)
        
        # Generate remaining tokens one by one
        for _ in range(max_new_tokens - 1):
            if next_token == self.eos_token_id:
                break
            
            # Inference with single token
            input_array = (ctypes.c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                input_array,
                1
            )
            tokens.append(next_token)
        
        return tokens
