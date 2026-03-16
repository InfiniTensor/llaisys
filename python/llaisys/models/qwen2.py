from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType

from pathlib import Path
import safetensors
import ctypes
import numpy as np
import struct
import json
import mmap
import os

from ..libllaisys import(
     DeviceType,
     LlaisysQwen2Meta,
     llaisys_qwen2_create,
     llaisys_qwen2_load_weight
)
TYPE_MAP={
         # 必须与 C++ enum llaisysDataType_t 完全一致
         # 参见 include/llaisys.h 和 python/llaisys/libllaisys/llaisys_types.py
         "F32":13,
         "F16":12,   # 修复: 之前错误地映射为 11(F8)
         "BF16":19
    }
class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.lib=LIB_LLAISYS

        config_path=os.path.join(model_path,"config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
        with open(config_path,"r") as f:
            config=json.load(f)
        
        meta=LlaisysQwen2Meta()
        meta.nlayer=config.get("num_hidden_layers",28)
        meta.hs=config.get("hidden_size",1536)
        meta.nh=config.get("num_attention_heads",12)
        meta.nkvh=config.get("num_key_value_heads",2)
        meta.vocab_size=config.get("vocab_size",151936)
        meta.maxseq=config.get("max_position_embeddings",32768)
        meta.epsilon=config.get("rms_norm_eps",1e-6)
        meta.theta=config.get("rope_theta",10000.0)

        config_dtype_str=config.get("torch_dtype","float16")

        target_key="F16"
        if config_dtype_str=="float32":
            target_key="F32"
        elif config_dtype_str=="bfloat16":
            target_key="BF16"
        elif config_dtype_str=="float16":
            target_key="F16"
        
        if target_key not in TYPE_MAP:
            print(f"Warning: Unknown dtype {config_dtype_str}, using F16")
            target_dtype=TYPE_MAP["F16"]
        else:
            target_dtype=TYPE_MAP[target_key]
        self.model=self.lib.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            device.value,
            None,
            0,
            target_dtype
        )

        self.lib.llaisysQwen2ModelForward.restype=ctypes.c_void_p
        self.lib.llaisysQwen2ModelForward.argtypes=[
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_size_t
        ]
        if hasattr(self.lib,'llaisysQwen2Sample'):
            self.lib.llaisysQwen2Sample.restype=ctypes.c_int
            self.lib.llaisysQwen2Sample.argtypes=[ctypes.c_void_p]

        model_path = Path(model_path)
        for file in sorted(model_path.glob("*.safetensors")):
            with open(file,'rb') as f_obj:
                header_size=struct.unpack('<Q',f_obj.read(8))[0]
                header_json=f_obj.read(header_size)
                header_data=json.loads(header_json)
                data_start=8+header_size

                with mmap.mmap(f_obj.fileno(),0,access=mmap.ACCESS_READ) as mm:
                    for name_,info in header_data.items():
                            if name_=="__metadata__": continue
                            dtype_str=info['dtype']
                            if dtype_str not in TYPE_MAP:
                                continue
                            dtype=TYPE_MAP[dtype_str]

                            shape=info['shape']
                            start,end=info['data_offsets']

                            np_dtype=np.float32 if dtype_str == 'F32' else np.uint16
                            itemsize=np.dtype(np_dtype).itemsize
                            tensor_np=np.frombuffer(
                                mm,
                                dtype=np_dtype,
                                count=(end-start)//itemsize,
                                offset=data_start+start
                            )

                            c_name=name_.encode('utf-8')
                            ndim=len(shape)
                            ShapeArrayType=ctypes.c_size_t*ndim
                            c_shape=ShapeArrayType(*shape)

                            c_data_ptr=ctypes.c_void_p(tensor_np.ctypes.data)
                            llaisys_qwen2_load_weight(
                                 self.model,
                                 c_name,
                                 c_data_ptr,
                                 c_shape,
                                 ndim,
                                 dtype
                            )
                            del tensor_np

    def forward(self,input_ids:Sequence[int],start_pos:int):
        seq_len=len(input_ids)

        InputArrayType=ctypes.c_int64*seq_len
        input_c_array=InputArrayType(*input_ids)

        input_ptr=ctypes.cast(input_c_array,ctypes.POINTER(ctypes.c_int64))

        logits_ptr=self.lib.llaisysQwen2ModelForward(
            self.model,
            input_ptr,
            seq_len,
            start_pos
        )
        return logits_ptr
    
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens=100
        
        tokens=list(inputs)
        start_pos=0

        eos_token_id=151643

        logits_ptr=self.forward(tokens,start_pos)

        next_token=self.lib.llaisysQwen2Sample(logits_ptr)
        tokens.append(next_token)
        if next_token ==eos_token_id:
            return tokens

        start_pos=len(inputs)

        for i in range(max_new_tokens - 1):
            input_step=[tokens[-1]]

            logits_ptr=self.forward(input_step,start_pos)

            next_token=self.lib.llaisysQwen2Sample(logits_ptr)

            tokens.append(next_token)
            # 与 HF 一样，将 EOS token 也包含在输出序列中
            if next_token == eos_token_id:
                break

            start_pos+=1

        return tokens
