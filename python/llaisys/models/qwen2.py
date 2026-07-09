import json
import ctypes
import numpy as np
import gc
import platform
import os
from typing import Sequence, List
from pathlib import Path
from ..libllaisys import LIB_LLAISYS, DeviceType, LlaisysQwen2Meta

# ==========================================
# 【核心修改】：精准定义与 C++ 交互的参数类型
# ==========================================
LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [
    ctypes.c_void_p,                # model handle
    ctypes.POINTER(ctypes.c_int64), # token_ids
    ctypes.c_size_t,                # ntoken
    ctypes.c_float,                 # temperature
    ctypes.c_float,                 # top_p
    ctypes.c_size_t                 # top_k
]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64
LIB_LLAISYS.llaisysQwen2ModelGetLogits.restype = ctypes.POINTER(ctypes.c_float)

try:
    from safetensors import safe_open
except ImportError:
    raise ImportError("pip install safetensors")

class Qwen2:
    def __init__(self, model_path: str, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = 0 
        self.meta.nlayer = cfg["num_hidden_layers"]
        self.meta.hs = cfg["hidden_size"]
        self.meta.nh = cfg["num_attention_heads"]
        self.meta.nkvh = cfg["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = cfg["intermediate_size"]
        
        raw_maxseq = cfg.get("max_position_embeddings", 4096)
        is_windows_ci = (platform.system() == "Windows" and os.environ.get("GITHUB_ACTIONS") == "true")
        
        if is_windows_ci:
            print(f"[CI-Optimization] Windows detected. Clamping max_seq from {raw_maxseq} to 1024 to save memory.")
            self.meta.maxseq = 1024
        else:
            self.meta.maxseq = raw_maxseq

        self.meta.voc = cfg["vocab_size"]
        self.meta.epsilon = cfg["rms_norm_eps"]
        self.meta.theta = cfg.get("rope_theta", 1000000.0)
        self.meta.end_token = cfg.get("eos_token_id", 151643)
        if isinstance(self.meta.end_token, list): self.meta.end_token = self.meta.end_token[0]

        print(f"[LLaisys] Init Qwen2: {self.meta.nlayer}L, {self.meta.hs}H, Context: {self.meta.maxseq}")

        self.handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), 
            device.value, 
            None, 
            0
        )
        if not self.handle: raise RuntimeError("Failed to create model")
        
        self.c_weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.handle).contents
        self._load_weights(model_path)

    def _load_weights(self, path):
       # 注意：虽然这里导入模型参数用了 PyTorch，但这不属于推理过程，
       # 只是为了读取 .safetensors 权重。项目规则是“推理阶段不使用PyTorch”。
        import torch 
        weight_files = sorted(list(path.glob("*.safetensors")))
        for f in weight_files:
            print(f"Loading {f.name}...")
            with safe_open(f, framework="pt", device="cpu") as st:
                for name in st.keys():
                    ptr = self._route(name)
                    if ptr:
                        tensor = st.get_tensor(name)
                        tensor = tensor.to(torch.float32)
                        data = tensor.numpy()
                        data = np.ascontiguousarray(data)
                        dst = LIB_LLAISYS.llaisysTensorData(ptr)
                        if dst: 
                            ctypes.memmove(dst, data.ctypes.data, data.nbytes)
                        del tensor
                        del data
            gc.collect()

    def _route(self, name):
        w = self.c_weights
        if name == "model.embed_tokens.weight": return w.in_embed
        if name == "model.norm.weight": return w.out_norm_w
        if name == "lm_head.weight": return w.out_embed
        
        if name.startswith("model.layers."):
            parts = name.split(".")
            idx = int(parts[2])
            if idx >= self.meta.nlayer: return None
            
            module = parts[3]
            sub = parts[4]
            is_bias = "bias" in parts[-1]

            if module == "self_attn":
                if sub == "q_proj": return w.attn_q_b[idx] if is_bias else w.attn_q_w[idx]
                if sub == "k_proj": return w.attn_k_b[idx] if is_bias else w.attn_k_w[idx]
                if sub == "v_proj": return w.attn_v_b[idx] if is_bias else w.attn_v_w[idx]
                if sub == "o_proj": return w.attn_o_w[idx]
            elif module == "mlp":
                if sub == "gate_proj": return w.mlp_gate_w[idx]
                if sub == "up_proj": return w.mlp_up_w[idx]
                if sub == "down_proj": return w.mlp_down_w[idx]
            elif module == "input_layernorm": return w.attn_norm_w[idx]
            elif module == "post_attention_layernorm": return w.mlp_norm_w[idx]
        return None

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.handle)

    def generate(self, inputs: Sequence[int], max_new_tokens=50, temperature=0.7, top_p=0.9, top_k=50, **kwargs) -> List[int]:
        curr = list(inputs)
        
        c_temp = ctypes.c_float(temperature)
        c_top_p = ctypes.c_float(top_p)
        c_top_k = ctypes.c_size_t(top_k)

        # Prefill 阶段
        seq_len = len(curr)
        arr = (ctypes.c_int64 * seq_len)(*curr)
        
        next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, arr, seq_len, c_temp, c_top_p, c_top_k)
        
        for _ in range(max_new_tokens):
            # 🚀 【核心修复】：先存进去！不管是不是结束符，都得保证它在列表里
            curr.append(next_tok)
            
            # 🚀 然后再判断要不要跳出循环
            if next_tok == self.meta.end_token or next_tok == 151645: 
                break
                
            # Decode 阶段：每次只传1个长度，返回下1个
            arr = (ctypes.c_int64 * 1)(next_tok)
            next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, arr, 1, c_temp, c_top_p, c_top_k)
            
        return curr

    def stream_generate(self, inputs: Sequence[int], max_new_tokens=400, temperature=0.7, top_p=0.9, top_k=50, **kwargs):
        """流式生成器：极其简洁，因为重活都交给 C++ 的 sampler 了！"""
        curr = list(inputs)
        
        c_temp = ctypes.c_float(temperature)
        c_top_p = ctypes.c_float(top_p)
        c_top_k = ctypes.c_size_t(top_k)

        # Prefill 阶段：喂给 C++ 整个问题，得到第一个回答的 Token
        seq_len = len(curr)
        arr = (ctypes.c_int64 * seq_len)(*curr)
        next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, arr, seq_len, c_temp, c_top_p, c_top_k)
        
        # 持续生成
        for _ in range(max_new_tokens):
            yield next_tok
            
            if next_tok == self.meta.end_token or next_tok == 151645: 
                break
                
            curr.append(next_tok)
            
            # Decode 阶段：把刚生成的词喂给 C++，换取下一个词
            arr = (ctypes.c_int64 * 1)(next_tok)
            next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, arr, 1, c_temp, c_top_p, c_top_k)