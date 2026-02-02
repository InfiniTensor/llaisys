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
         "F32":13,
         "F16":11,
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
        meta.max_seq_len=config.get("max_position_embeddings",32768)
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
        print(f"Python: Loading Model config | Layers={meta.nlayer} | HS={meta.hs} | Dtype={target_key}({target_dtype})")
        print("Python:calling the cpp to create model ...")
        self.model=self.lib.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            device.value,
            None,
            0,
            target_dtype
        )
        print("Python: Model handle received:",self.model)

        model_path = Path(model_path)

        print("Python: Scanning .safetensors files...")

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
                    print("Python: All weights loaded")
                    pass
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function

        return []
