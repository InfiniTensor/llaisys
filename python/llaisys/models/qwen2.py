import ctypes
import numpy as np
import gc
from enum import IntEnum
from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import *
from ..tensor import Tensor
import torch


from pathlib import Path
import safetensors
import json


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        self._load_config()
        self._load_weights()
       
    def __delete__(self):
        LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        ptr = np.array(inputs, dtype=np.int64).ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        l = len(inputs)
        ret = list(inputs)
        id = 0
        while id != self.config["eos_token_id"]:
            id = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self.model, ptr, ctypes.c_size_t(l)))
            ret.append(id)
            ptr = ctypes.byref(ctypes.c_int64(id))
            l = 1
        return ret
    
    def _load_config(self):
        config_file = self.model_path / "config.json"
        with open(config_file, "r") as f:
            self.config = json.load(f)
        meta = Qwen2Meta()
        meta.dtype = ctypes.c_int(DataType.BF16)
        meta.nlayer = ctypes.c_size_t(self.config["num_hidden_layers"])
        meta.hs = ctypes.c_size_t(self.config["hidden_size"])
        meta.nh = ctypes.c_size_t(self.config["num_attention_heads"])
        meta.nkvh = ctypes.c_size_t(self.config["num_key_value_heads"])
        meta.dh = ctypes.c_size_t(self.config["hidden_size"] // self.config["num_attention_heads"])
        meta.di = ctypes.c_size_t(self.config["intermediate_size"])
        meta.maxseq = ctypes.c_size_t(self.config["max_position_embeddings"])
        meta.voc = ctypes.c_size_t(self.config["vocab_size"])
        meta.epsilon = ctypes.c_float(self.config["rms_norm_eps"])
        meta.theta = ctypes.c_float(self.config["rope_theta"])
        meta.end_token = ctypes.c_int64(self.config["eos_token_id"])


        id = ctypes.c_int(0)
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(ctypes.byref(meta), self.device, ctypes.byref(id), 1)

    def _load_weights(self):
        for file in sorted(self.model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="torch", device="cpu")
            for name_ in data_.keys():
                tensor = data_.get_tensor(name_)
                name_c = ctypes.c_char_p(name_.encode('utf-8'))
                LIB_LLAISYS.llaisysQwen2modelLoadWeight(self.model, ctypes.c_void_p(tensor.data_ptr()), name_c)
                del tensor
        gc.collect()