from typing import Sequence, List
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta

from pathlib import Path
from ctypes import c_int, c_int64, c_size_t, POINTER, byref
import json
import safetensors
import torch
import numpy as np


class Qwen2TP:
    """Tensor Parallel Qwen2 Model.
    
    This class distributes the model across multiple GPUs using tensor parallelism.
    """

    def __init__(self, model_path, device_ids: List[int], device: DeviceType = DeviceType.NVIDIA):
        """
        Args:
            model_path: Path to the model directory
            device_ids: List of GPU device IDs to use for tensor parallelism
            device: Device type (must be NVIDIA for tensor parallelism)
        """
        if device != DeviceType.NVIDIA:
            raise ValueError("Tensor parallelism is only supported on NVIDIA GPUs")
        
        if len(device_ids) < 2:
            raise ValueError("Tensor parallelism requires at least 2 GPUs")

        model_path = Path(model_path)
        self._device_ids = device_ids
        self._world_size = len(device_ids)
        self._device = device

        # Load config
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract model parameters
        hidden_size = config["hidden_size"]
        num_attention_heads = config["num_attention_heads"]
        num_key_value_heads = config["num_key_value_heads"]
        head_dim = hidden_size // num_attention_heads
        intermediate_size = config["intermediate_size"]

        # Validate divisibility
        if num_attention_heads % self._world_size != 0:
            raise ValueError(f"num_attention_heads ({num_attention_heads}) must be divisible by world_size ({self._world_size})")
        if num_key_value_heads % self._world_size != 0:
            raise ValueError(f"num_key_value_heads ({num_key_value_heads}) must be divisible by world_size ({self._world_size})")
        if intermediate_size % self._world_size != 0:
            raise ValueError(f"intermediate_size ({intermediate_size}) must be divisible by world_size ({self._world_size})")

        # Create meta structure
        self._meta = LlaisysQwen2Meta()
        self._meta.dtype = DataType.BF16.value
        self._meta.nlayer = config["num_hidden_layers"]
        self._meta.hs = hidden_size
        self._meta.nh = num_attention_heads
        self._meta.nkvh = num_key_value_heads
        self._meta.dh = head_dim
        self._meta.di = intermediate_size
        self._meta.maxseq = min(config.get("max_position_embeddings", 131072), 4096)
        self._meta.voc = config["vocab_size"]
        self._meta.epsilon = config.get("rms_norm_eps", 1e-6)
        self._meta.theta = config.get("rope_theta", 10000.0)
        self._meta.end_token = config.get("eos_token_id", 151643)

        self._nlayer = self._meta.nlayer

        # Create TP model
        device_ids_arr = (c_int * self._world_size)(*device_ids)
        self._model = LIB_LLAISYS.llaisysQwen2ModelTPCreate(
            byref(self._meta),
            device_ids_arr,
            self._world_size
        )

        # Get weights for each rank
        self._weights_per_rank = []
        for i in range(self._world_size):
            weights_ptr = LIB_LLAISYS.llaisysQwen2ModelTPWeights(self._model, i)
            self._weights_per_rank.append(weights_ptr.contents)

        # Load weights from safetensors
        self._load_weights(model_path)
        
        # Flag for automatic warm-up on first generate call
        self._needs_warmup = True

    def _warmup(self):
        """Run a 1-token warm-up inference to initialize NCCL internal state.
        
        This prevents 'unhandled cuda error' on the first real inference.
        The warm-up runs a minimal forward pass that triggers all NCCL collectives.
        """
        # Run 1-token inference (token 1 is a safe choice)
        warmup_arr = (c_int64 * 1)(1)
        LIB_LLAISYS.llaisysQwen2ModelTPInfer(
            self._model,
            warmup_arr,
            c_size_t(1)
        )
        # Reset cache so warm-up doesn't affect subsequent inference
        LIB_LLAISYS.llaisysQwen2ModelTPResetCache(self._model)

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelTPDestroy(self._model)
            self._model = None

    def _load_weights(self, model_path: Path):
        """Load and shard weights from safetensors files."""
        
        # Collect all tensors from safetensors files
        all_tensors = {}
        for file in sorted(model_path.glob("*.safetensors")):
            data = safetensors.safe_open(file, framework="pt", device="cpu")
            for name in data.keys():
                all_tensors[name] = data.get_tensor(name)

        nh = self._meta.nh
        nkvh = self._meta.nkvh
        dh = self._meta.dh
        di = self._meta.di
        hs = self._meta.hs
        world_size = self._world_size

        # Sharded dimensions
        nh_shard = nh // world_size
        nkvh_shard = nkvh // world_size
        di_shard = di // world_size

        # Helper to load shared tensor (replicated on all ranks)
        def load_shared(tensor_handle, tensor_data, dtype=torch.bfloat16):
            data = tensor_data.to(dtype).contiguous()
            data_ptr = data.data_ptr()
            LIB_LLAISYS.tensorLoad(tensor_handle, data_ptr)

        # Helper to column-shard a weight tensor
        def load_column_sharded(weight_handles, tensor_data, output_dim_size, dtype=torch.bfloat16):
            """Column-shard a weight tensor along the output dimension."""
            for rank in range(world_size):
                start = rank * output_dim_size
                end = start + output_dim_size
                shard = tensor_data[start:end].to(dtype).contiguous()
                LIB_LLAISYS.tensorLoad(weight_handles[rank], shard.data_ptr())

        # Helper to row-shard a weight tensor
        def load_row_sharded(weight_handles, tensor_data, input_dim_size, dtype=torch.bfloat16):
            """Row-shard a weight tensor along the input dimension."""
            for rank in range(world_size):
                start = rank * input_dim_size
                end = start + input_dim_size
                shard = tensor_data[:, start:end].to(dtype).contiguous()
                LIB_LLAISYS.tensorLoad(weight_handles[rank], shard.data_ptr())

        # Load embedding weights (shared)
        for rank in range(world_size):
            load_shared(self._weights_per_rank[rank].in_embed, 
                       all_tensors["model.embed_tokens.weight"])
            load_shared(self._weights_per_rank[rank].out_embed, 
                       all_tensors["lm_head.weight"])
            load_shared(self._weights_per_rank[rank].out_norm_w, 
                       all_tensors["model.norm.weight"])

        # Load per-layer weights
        for i in range(self._nlayer):
            prefix = f"model.layers.{i}."

            for rank in range(world_size):
                w = self._weights_per_rank[rank]

                # Layer norm weights (shared)
                load_shared(w.attn_norm_w[i],
                           all_tensors[prefix + "input_layernorm.weight"])
                load_shared(w.mlp_norm_w[i],
                           all_tensors[prefix + "post_attention_layernorm.weight"])

            # Column-shard Q, K, V weights and biases
            # attn_q_w: [nh * dh, hs] -> [nh_shard * dh, hs] per rank
            load_column_sharded(
                [w.attn_q_w[i] for w in self._weights_per_rank],
                all_tensors[prefix + "self_attn.q_proj.weight"],
                nh_shard * dh
            )
            load_column_sharded(
                [w.attn_q_b[i] for w in self._weights_per_rank],
                all_tensors[prefix + "self_attn.q_proj.bias"],
                nh_shard * dh
            )

            # attn_k_w: [nkvh * dh, hs] -> [nkvh_shard * dh, hs] per rank
            load_column_sharded(
                [w.attn_k_w[i] for w in self._weights_per_rank],
                all_tensors[prefix + "self_attn.k_proj.weight"],
                nkvh_shard * dh
            )
            load_column_sharded(
                [w.attn_k_b[i] for w in self._weights_per_rank],
                all_tensors[prefix + "self_attn.k_proj.bias"],
                nkvh_shard * dh
            )

            # attn_v_w: [nkvh * dh, hs] -> [nkvh_shard * dh, hs] per rank
            load_column_sharded(
                [w.attn_v_w[i] for w in self._weights_per_rank],
                all_tensors[prefix + "self_attn.v_proj.weight"],
                nkvh_shard * dh
            )
            load_column_sharded(
                [w.attn_v_b[i] for w in self._weights_per_rank],
                all_tensors[prefix + "self_attn.v_proj.bias"],
                nkvh_shard * dh
            )

            # Row-shard O projection weight
            # attn_o_w: [hs, nh * dh] -> [hs, nh_shard * dh] per rank
            load_row_sharded(
                [w.attn_o_w[i] for w in self._weights_per_rank],
                all_tensors[prefix + "self_attn.o_proj.weight"],
                nh_shard * dh
            )

            # Column-shard MLP gate and up weights
            # mlp_gate_w: [di, hs] -> [di_shard, hs] per rank
            load_column_sharded(
                [w.mlp_gate_w[i] for w in self._weights_per_rank],
                all_tensors[prefix + "mlp.gate_proj.weight"],
                di_shard
            )
            load_column_sharded(
                [w.mlp_up_w[i] for w in self._weights_per_rank],
                all_tensors[prefix + "mlp.up_proj.weight"],
                di_shard
            )

            # Row-shard MLP down weight
            # mlp_down_w: [hs, di] -> [hs, di_shard] per rank
            load_row_sharded(
                [w.mlp_down_w[i] for w in self._weights_per_rank],
                all_tensors[prefix + "mlp.down_proj.weight"],
                di_shard
            )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """Generate tokens using tensor parallel inference."""
        # Note: Currently only supports greedy decoding (top_k=1)
        # Temperature and top_p are ignored for simplicity
        
        # Automatic warm-up on first call to initialize NCCL state
        if self._needs_warmup:
            self._warmup()
            self._needs_warmup = False
        
        # Reset KV cache for new generation
        LIB_LLAISYS.llaisysQwen2ModelTPResetCache(self._model)

        # Convert input to ctypes array
        input_len = len(inputs)
        input_arr = (c_int64 * input_len)(*inputs)

        # Output tokens list (starts with input)
        output_tokens = list(inputs)

        # First forward pass with all input tokens
        next_token = LIB_LLAISYS.llaisysQwen2ModelTPInfer(
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
            next_token = LIB_LLAISYS.llaisysQwen2ModelTPInfer(
                self._model,
                single_token,
                c_size_t(1)
            )
            output_tokens.append(next_token)

        return output_tokens

    @property
    def world_size(self) -> int:
        """Get the tensor parallel world size."""
        return self._world_size

    @property
    def device_ids(self) -> List[int]:
        """Get the list of GPU device IDs."""
        return self._device_ids.copy()
