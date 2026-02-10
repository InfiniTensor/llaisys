import ctypes
import json
import os
import mmap
import struct
from typing import List, Dict, Optional, Sequence, Any
import numpy as np
from pathlib import Path

from ..libllaisys import LIB_LLAISYS, llaisysTensor_t, llaisysDataType_t, llaisysDeviceType_t, DataType, DeviceType
from ..libllaisys.models import LlaisysQwen2Meta, LlaisysQwen2Weights, LlaisysQwen2Model, llaisysQwen2ModelHandle
from ..tensor import Tensor

LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [ctypes.POINTER(LlaisysQwen2Meta), llaisysDeviceType_t, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = llaisysQwen2ModelHandle

LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2ModelHandle]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2ModelHandle]
LIB_LLAISYS.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [llaisysQwen2ModelHandle, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64


class Qwen2:
    def __init__(self, model_path: str, device: DeviceType = DeviceType.CPU, device_id: int = 0):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.device = device
        self.device_id = device_id
        
        self.meta = LlaisysQwen2Meta()
        
        dtype_str = config.get("torch_dtype", "float32")
        self.meta.dtype = DataType.F32 

        self.meta.nlayer = config.get("num_hidden_layers", 24)
        self.meta.hs = config.get("hidden_size", 2048)
        self.meta.nh = config.get("num_attention_heads", 16)
        self.meta.nkvh = config.get("num_key_value_heads", 16)
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = config.get("intermediate_size", 11008)
        self.meta.maxseq = config.get("max_position_embeddings", 8192)
        self.meta.voc = config.get("vocab_size", 151936)
        self.meta.epsilon = config.get("rms_norm_eps", 1e-6)
        self.meta.theta = config.get("rope_theta", 1000000.0)
        self.meta.end_token = 151643 # Placeholder
        
        dev_ids = (ctypes.c_int * 1)(device_id)
        self.handle = LIB_LLAISYS.llaisysQwen2ModelCreate(ctypes.byref(self.meta), device, dev_ids, 1)
        self.weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.handle)
        self.tensors_ref = []

        for file in sorted(model_path.glob("*.safetensors")):
            print(f"Loading weights from {file}...")
            weights_data = self._load_safetensors_bf16_as_f32(file)
            for key, arr in weights_data.items():
                if not arr.flags['C_CONTIGUOUS']:
                    arr = np.ascontiguousarray(arr)
                
                t = Tensor(list(arr.shape), self.meta.dtype, device, device_id)
                t.load(ctypes.c_void_p(arr.ctypes.data))
                
                self._assign_weight(key, t)

    def _load_safetensors_bf16_as_f32(self, path: Path) -> Dict[str, np.ndarray]:
        tensors = {}
        with open(path, 'rb') as f:
            length_bytes = f.read(8)
            if not length_bytes: return {}
            header_size = struct.unpack('<Q', length_bytes)[0]
            
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes)
            
            fileno = f.fileno()
            total_size = os.fstat(fileno).st_size
            mm = mmap.mmap(fileno, total_size, access=mmap.ACCESS_READ)
            
            data_start = 8 + header_size
            
            for key, info in header.items():
                if key == "__metadata__": continue
                
                dtype_str = info['dtype']
                shape = info['shape']
                start, end = info['data_offsets']
                
                abs_start = data_start + start
                
                if dtype_str == "BF16" or dtype_str == "bfloat16":
                    raw_u16 = np.frombuffer(mm, dtype=np.uint16, count=(end-start)//2, offset=abs_start)
                    u32 = raw_u16.astype(np.uint32) << 16
                    del raw_u16
                    arr = u32.view(np.float32).reshape(shape)
                    tensors[key] = arr
                elif dtype_str == "F32" or dtype_str == "float32":
                    raw_f32 = np.frombuffer(mm, dtype=np.float32, count=(end-start)//4, offset=abs_start)
                    arr = np.array(raw_f32).reshape(shape)
                    del raw_f32
                    tensors[key] = arr
                else: 
                     if dtype_str == "F16" or dtype_str == "float16":
                         raw_f16 = np.frombuffer(mm, dtype=np.float16, count=(end-start)//2, offset=abs_start)
                         arr = raw_f16.astype(np.float32).reshape(shape)
                         del raw_f16
                         tensors[key] = arr
            
            mm.close()
            
        return tensors

    def _assign_weight(self, name: str, t: Tensor):
        w = self.weights_ptr.contents
        self.tensors_ref.append(t)
        
        if name == "model.embed_tokens.weight":
             w.in_embed = t.lib_tensor()
        elif name == "lm_head.weight":
             w.out_embed = t.lib_tensor()
        elif name == "model.norm.weight":
             w.out_norm_w = t.lib_tensor()
        elif name.startswith("model.layers."):
            parts = name.split(".")
            layer_idx = int(parts[2])
            suffix = ".".join(parts[3:])
            
            def set_w(target_ptr):
                target_ptr[layer_idx] = t.lib_tensor()

            if suffix == "input_layernorm.weight":
                set_w(w.attn_norm_w)
            elif suffix == "self_attn.q_proj.weight":
                set_w(w.attn_q_w)
            elif suffix == "self_attn.q_proj.bias":
                set_w(w.attn_q_b)
            elif suffix == "self_attn.k_proj.weight":
                set_w(w.attn_k_w)
            elif suffix == "self_attn.k_proj.bias":
                set_w(w.attn_k_b)
            elif suffix == "self_attn.v_proj.weight":
                set_w(w.attn_v_w)
            elif suffix == "self_attn.v_proj.bias":
                set_w(w.attn_v_b)
            elif suffix == "self_attn.o_proj.weight":
                set_w(w.attn_o_w)
            elif suffix == "post_attention_layernorm.weight":
                set_w(w.mlp_norm_w)
            elif suffix == "mlp.gate_proj.weight":
                set_w(w.mlp_gate_w)
            elif suffix == "mlp.up_proj.weight":
                set_w(w.mlp_up_w)
            elif suffix == "mlp.down_proj.weight":
                set_w(w.mlp_down_w)

    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.handle)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 20,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> List[int]:
        
        generated = []
        tokens = list(inputs)
        
        arr = (ctypes.c_int64 * len(tokens))(*tokens)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, arr, len(tokens))
        generated.append(next_token)
        tokens = [next_token] 
        
        for _ in range(max_new_tokens - 1):
             arr = (ctypes.c_int64 * 1)(*tokens)
             next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, arr, 1)
             generated.append(next_token)
             tokens = [next_token]
             
             if next_token == self.meta.end_token:
                 break
        
        return list(inputs) + generated
