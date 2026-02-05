from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType

from .qwen2_binding import register_qwen2_lib, Qwen2MetaCStruct

from pathlib import Path
import safetensors.torch
import torch

import os
import json
import ctypes

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.lib = LIB_LLAISYS
        register_qwen2_lib(self.lib) # Register C functions

        model_path = Path(model_path)
        config_path = model_path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        
        with open(config_path, "r") as f:
            config = json.load(f)

        self.meta = Qwen2MetaCStruct()
        
        # Populate Meta (Default fallback values based on typical Qwen2 config)
        self.meta.hs = config.get("hidden_size", 1536)
        self.meta.nlayer = config.get("num_hidden_layers", 28)
        self.meta.nh = config.get("num_attention_heads", 12)
        self.meta.nkvh = config.get("num_key_value_heads", 2)
        self.meta.voc = config.get("vocab_size", 151936)
        self.meta.maxseq = config.get("max_position_embeddings", 32768)
        self.meta.di = config.get("intermediate_size", 8960)
        self.meta.epsilon = config.get("rms_norm_eps", 1e-6)
        self.meta.theta = config.get("rope_theta", 10000.0)
        self.meta.dh = self.meta.hs // self.meta.nh
        
        # Determine EOS token
        eos_id = config.get("eos_token_id", 151643) # ID of <|endoftext|>
        if isinstance(eos_id, list):
             self.meta.end_token = eos_id[0]
        else:
             self.meta.end_token = eos_id
             
        # Set dtype for the model struct (match weight dtype when possible)
        torch_dtype = str(config.get("torch_dtype", "float32")).lower()
        if "bfloat16" in torch_dtype or "bf16" in torch_dtype:
            self.meta.dtype = DataType.BF16
        elif "float16" in torch_dtype or "fp16" in torch_dtype:
            self.meta.dtype = DataType.F16
        else:
            self.meta.dtype = DataType.F32

        # Create C Model
        device_ids = (ctypes.c_int * 1)(0)
        # Use F32 for KV cache for stability on CPU
        self.model = self.lib.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            device,
            device_ids,
            1
        )

        if not self.model:
            raise RuntimeError("Failed to create native Qwen2 model instance.")

        # Load Weights
        for file in sorted(model_path.glob("*.safetensors")):
            print(f"Loading weights from {file}...")
            # Use safe_open from safetensors (torch backend to support BF16)
            with safetensors.torch.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    
                    # Map torch dtype to Llaisys DataType
                    dt = DataType.F32
                    if tensor.dtype == torch.float16:
                        dt = DataType.F16
                    elif tensor.dtype == torch.float32:
                        dt = DataType.F32
                    elif tensor.dtype == torch.bfloat16:
                        dt = DataType.BF16
                    elif tensor.dtype == torch.int64:
                        dt = DataType.I64
                    
                    # Ensure contiguous
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                        
                    shape = tensor.shape
                    c_shape = (ctypes.c_size_t * len(shape))(*shape)
                    
                    # Keep a reference to data pointer valid during the C call
                    data_ptr = ctypes.c_void_p(tensor.data_ptr())

                    self.lib.llaisysQwen2LoadWeight(
                        self.model,
                        name.encode('utf-8'),
                        data_ptr,
                        c_shape,
                        len(shape),
                        dt
                    )

    def __del__(self):
        if hasattr(self, 'model') and self.model:
            self.lib.llaisysQwen2ModelDestroy(self.model)
            self.model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 20
            
        tokens = list(inputs)
        start_pos = 0
        
        for _ in range(max_new_tokens):
            if start_pos == 0:
                current_input = tokens
            else:
                current_input = tokens[-1:] # Next token generation: use only last token
                
            n_tokens = len(current_input)
            c_inputs = (ctypes.c_int64 * n_tokens)(*current_input)
            
            # Infer (argmax inside backend)
            next_token_id = self.lib.llaisysQwen2ModelInfer(
                self.model,
                c_inputs,
                n_tokens,
                start_pos
            )
            
            tokens.append(next_token_id)
            start_pos += n_tokens
            
            if next_token_id == self.meta.end_token:
                break
                
        return tokens
