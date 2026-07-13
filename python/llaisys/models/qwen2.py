from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys.models import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor
import ctypes
from ctypes import POINTER, c_int, c_int64, c_float, c_size_t
from pathlib import Path
import json
import numpy as np
import mmap
import struct

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # 1. Load Config
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        # 2. Prepare Meta
        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = 19 # BF16
        self.meta.nlayer = config["num_hidden_layers"]
        self.meta.hs = config["hidden_size"]
        self.meta.nh = config["num_attention_heads"]
        self.meta.nkvh = config["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = config["intermediate_size"]
        self.meta.maxseq = 2048
        self.meta.voc = config["vocab_size"]
        self.meta.epsilon = config["rms_norm_eps"]
        self.meta.theta = config.get("rope_theta", 10000.0)
        self.meta.end_token = 151643 # <|end_of_text|>

        # 3. Create C Model
        device_ids = (c_int * 1)(0)
        self._model_handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            device.value,
            device_ids,
            1
        )
        
        # 4. Get Weights Structure Pointers
        self.weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model_handle).contents

        # 5. Load Weights Manually
        print("Loading weights...")
        self._load_safetensors_manually(model_path)

    def _load_safetensors_manually(self, model_path: Path):
        """
        Manually parse safetensors headers and mmap data as uint16.
        """
        for file in sorted(model_path.glob("*.safetensors")):
            with open(file, 'rb') as f:
                # Read Header Size
                header_len_bytes = f.read(8)
                if not header_len_bytes: continue
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                
                # Read Header
                header_bytes = f.read(header_len)
                header = json.loads(header_bytes)
                
                # Start of data section
                data_start = 8 + header_len
                
                # Use mmap
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    for name, info in header.items():
                        if name == "__metadata__": continue
                        
                        target_tensor = self._map_weight_name(name)
                        if not target_tensor:
                            continue
                        
                        begin, end = info['data_offsets']
                        total_bytes = end - begin
                        
                        # Create numpy view
                        raw_data = np.frombuffer(
                            mm, 
                            dtype=np.uint16, 
                            count=total_bytes // 2, 
                            offset=data_start + begin
                        )
                        
                        target_tensor.load(raw_data.ctypes.data)
                        
                        del raw_data 

    def _map_weight_name(self, name):
        if "model.embed_tokens.weight" in name:
            return Tensor(tensor=self.weights_ptr.in_embed)
        elif "lm_head.weight" in name:
            return Tensor(tensor=self.weights_ptr.out_embed)
        elif "model.norm.weight" in name:
            return Tensor(tensor=self.weights_ptr.out_norm_w)
        elif "layers" in name:
            parts = name.split(".")
            layer_idx = int(parts[2])
            module = parts[3]
            
            if module == "input_layernorm":
                return Tensor(tensor=self.weights_ptr.attn_norm_w[layer_idx])
            elif module == "post_attention_layernorm":
                return Tensor(tensor=self.weights_ptr.mlp_norm_w[layer_idx])
            elif module == "self_attn":
                proj = parts[4]
                type_ = parts[5]
                if type_ == "weight":
                    if "q_proj" in proj: return Tensor(tensor=self.weights_ptr.attn_q_w[layer_idx])
                    elif "k_proj" in proj: return Tensor(tensor=self.weights_ptr.attn_k_w[layer_idx])
                    elif "v_proj" in proj: return Tensor(tensor=self.weights_ptr.attn_v_w[layer_idx])
                    elif "o_proj" in proj: return Tensor(tensor=self.weights_ptr.attn_o_w[layer_idx])
                elif type_ == "bias":
                    if "q_proj" in proj: return Tensor(tensor=self.weights_ptr.attn_q_b[layer_idx])
                    elif "k_proj" in proj: return Tensor(tensor=self.weights_ptr.attn_k_b[layer_idx])
                    elif "v_proj" in proj: return Tensor(tensor=self.weights_ptr.attn_v_b[layer_idx])
            elif module == "mlp":
                proj = parts[4]
                if "gate_proj" in proj: return Tensor(tensor=self.weights_ptr.mlp_gate_w[layer_idx])
                elif "up_proj" in proj: return Tensor(tensor=self.weights_ptr.mlp_up_w[layer_idx])
                elif "down_proj" in proj: return Tensor(tensor=self.weights_ptr.mlp_down_w[layer_idx])
        return None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        tokens = list(inputs)
        
        # 1. Prefill
        input_arr = (c_int64 * len(tokens))(*tokens)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model_handle,
            input_arr,
            len(tokens)
        )
        tokens.append(next_token)
        
        # 2. Decode Loop
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
                
            input_arr = (c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model_handle,
                input_arr,
                1
            )
            tokens.append(next_token)

        return tokens
    
    def __del__(self):
        try:
            if hasattr(self, '_model_handle') and self._model_handle:
                if LIB_LLAISYS and hasattr(LIB_LLAISYS, 'llaisysQwen2ModelDestroy'):
                    LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model_handle)
                    self._model_handle = None
        except:
            pass
