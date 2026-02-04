from typing import Sequence
import os
import json
import ctypes
import numpy as np
from pathlib import Path
from safetensors import safe_open
import torch

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..libllaisys import DataType

class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = DataType.F32.value
        self.meta.nlayer = cfg["num_hidden_layers"]
        self.meta.hs = cfg["hidden_size"]
        self.meta.nh = cfg["num_attention_heads"]
        self.meta.nkvh = cfg["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = cfg["intermediate_size"]
        self.meta.maxseq = 2048 
        self.meta.voc = cfg["vocab_size"]
        self.meta.epsilon = cfg["rms_norm_eps"]
        self.meta.theta = cfg.get("rope_theta", 10000.0)
        self.meta.end_token = 151643

        self.handle = LIB_LLAISYS.llaisysQwen2ModelCreate(ctypes.byref(self.meta), device.value, None, 0)
        self.weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.handle).contents

        for file in sorted(model_path.glob("*.safetensors")):
            print(f"Loading {file.name}...")
            with safe_open(file, framework="pt", device="cpu") as f:
                for name_ in f.keys():
                    # 🔴 修复：不再跳过 Bias
                    tensor_ = f.get_tensor(name_)
                    if tensor_.dtype != torch.float32:
                        tensor_ = tensor_.to(torch.float32)
                    self._load_weight(name_, tensor_.numpy())

    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.handle)

    def _load_weight(self, name, data):
        data_f32 = np.ascontiguousarray(data)
        src_ptr = data_f32.ctypes.data_as(ctypes.c_void_p)
        
        def copy_to(target_tensor_ptr):
            if not target_tensor_ptr: return
            LIB_LLAISYS.tensorLoad(target_tensor_ptr, src_ptr)

        w = self.weights
        
        # 1. 加载 Weight
        if name.endswith(".weight"):
            key = name[:-7] # 去掉 .weight
            if key == "model.embed_tokens": copy_to(w.in_embed)
            elif key == "model.norm": copy_to(w.out_norm_w)
            elif key == "lm_head": copy_to(w.out_embed)
            elif key.startswith("model.layers."):
                parts = key.split(".")
                idx = int(parts[2])
                module = parts[3]
                if module == "input_layernorm": copy_to(w.attn_norm_w[idx])
                elif module == "post_attention_layernorm": copy_to(w.mlp_norm_w[idx])
                elif module == "self_attn":
                    sub = parts[4]
                    if sub == "q_proj": copy_to(w.attn_q_w[idx])
                    elif sub == "k_proj": copy_to(w.attn_k_w[idx])
                    elif sub == "v_proj": copy_to(w.attn_v_w[idx])
                    elif sub == "o_proj": copy_to(w.attn_o_w[idx])
                elif module == "mlp":
                    sub = parts[4]
                    if sub == "gate_proj": copy_to(w.mlp_gate_w[idx])
                    elif sub == "up_proj": copy_to(w.mlp_up_w[idx])
                    elif sub == "down_proj": copy_to(w.mlp_down_w[idx])

        # 2. 加载 Bias (🔴 新增逻辑)
        elif name.endswith(".bias"):
            key = name[:-5] # 去掉 .bias
            if key.startswith("model.layers."):
                parts = key.split(".")
                idx = int(parts[2])
                module = parts[3]
                if module == "self_attn":
                    sub = parts[4]
                    if sub == "q_proj": copy_to(w.attn_q_b[idx])
                    elif sub == "k_proj": copy_to(w.attn_k_b[idx])
                    elif sub == "v_proj": copy_to(w.attn_v_b[idx])
                    # o_proj, mlp 等通常无 bias，忽略即可

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 100,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None: max_new_tokens = 100
        
        tokens = list(inputs)
        
        # Prefill
        in_arr = (ctypes.c_int64 * len(tokens))(*tokens)
        next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, in_arr, len(tokens))
        tokens.append(next_tok)
        
        # Decode
        for _ in range(max_new_tokens - 1):
            in_arr = (ctypes.c_int64 * 1)(tokens[-1])
            next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, in_arr, 1)
            tokens.append(next_tok)
            if next_tok == self.meta.end_token:
                break
        
        # 🔴 修改这里：返回完整的 tokens 列表，不要 [len(inputs):]
        return tokens