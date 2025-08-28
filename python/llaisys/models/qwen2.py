# python/llaisys/models/qwen2.py
from ctypes import c_size_t, c_longlong, c_void_p, POINTER
from pathlib import Path
import json
import numpy as np
import safetensors

# 统一从这里拿句柄和结构体
from ..libllaisys import LIB_LLAISYS, DeviceType               # 复用已加载的 DLL 句柄
from ..libllaisys.qwen2 import LlaisysQwen2Meta                # 你的 ctypes 结构体
from ..libllaisys.llaisys_types import DataType as DType       

HF2LLA = {"float32": DType.F32, "float16": DType.F16, "bfloat16": DType.BF16}

class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        mp = Path(model_path)
        cfg = json.loads((mp / "config.json").read_text(encoding="utf-8"))

        meta = LlaisysQwen2Meta()
        meta.dtype   = int(HF2LLA.get(cfg.get("torch_dtype", "float32"), DType.F32))
        meta.nlayer  = int(cfg["num_hidden_layers"])
        meta.hs      = int(cfg["hidden_size"])
        meta.nh      = int(cfg["num_attention_heads"])
        meta.nkvh    = int(cfg.get("num_key_value_heads", meta.nh))
        meta.dh      = meta.hs // meta.nh
        meta.di      = int(cfg["intermediate_size"])
        meta.maxseq  = int(cfg.get("max_position_embeddings", 4096))
        meta.voc     = int(cfg["vocab_size"])
        meta.epsilon = float(cfg.get("rms_norm_eps", 1e-6))
        meta.theta   = float(cfg.get("rope_theta", 10000.0))
        meta.end_token = int(cfg.get("eos_token_id", 2))

        self._handle = LIB_LLAISYS.llaisysQwen2ModelCreate(meta, int(device), None, 0)

        # numpy 路径支持的 dtype 映射
        NP_DTYPE_TO_LLA = {
            np.float32: int(DType.F32),
            np.float16: int(DType.F16),
            # np.uint16: 也用作 BF16，下方单独判断
        }   


        # 遍历 safetensors，把权重送进后端
        for file in sorted(mp.glob("*.safetensors")):
            # 先用 numpy 框架打开，能读就读；遇到 bfloat16 再走 torch 兜底
            with safetensors.safe_open(file, framework="numpy", device="cpu") as f_np:
                for name in f_np.keys():
                    try:
                        arr = f_np.get_tensor(name)  # 可能抛 "bfloat16 not understood"
                        # dtype 映射
                        if arr.dtype.type in NP_DTYPE_TO_LLA:
                            lla_dtype = NP_DTYPE_TO_LLA[arr.dtype.type]
                        elif str(arr.dtype) == "uint16":
                            # 有些包直接把 bf16 存成 uint16
                            lla_dtype = int(DType.BF16)
                        else:
                            raise TypeError(f"unsupported numpy dtype: {arr.dtype}")

                        shape = (c_size_t * arr.ndim)(*arr.shape)
                        ptr = c_void_p(arr.__array_interface__["data"][0])
                        rc = LIB_LLAISYS.llaisysQwen2ModelLoadNamedWeight(
                            self._handle, name.encode("utf-8"),
                            ptr, shape, c_size_t(arr.ndim), lla_dtype
                        )
                        if rc != 0:
                            raise RuntimeError(f"load weight failed: {name}")

                    except TypeError as e:
                        # numpy 不认 bfloat16：切换 torch 兜底
                        if "bfloat16" not in str(e).lower():
                            raise

                        import safetensors.torch as st
                        import torch
                        # 只取当前 name，避免整包复制太多内存
                        tdict = st.load_file(str(file), device="cpu")
                        t = tdict[name]  # torch.Tensor

                        if t.dtype == torch.bfloat16:
                            # 用原始位模式传入后端（uint16 数组）
                            arr_u16 = t.view(torch.uint16).cpu().numpy()
                            lla_dtype = int(DType.BF16)
                        elif t.dtype == torch.float16:
                            arr_u16 = t.cpu().numpy()
                            lla_dtype = int(DType.F16)
                        elif t.dtype == torch.float32:
                            arr_u16 = t.cpu().numpy()
                            lla_dtype = int(DType.F32)
                        else:
                            raise TypeError(f"unsupported torch dtype: {t.dtype}")

                        shape = (c_size_t * arr_u16.ndim)(*arr_u16.shape)
                        ptr = c_void_p(arr_u16.__array_interface__["data"][0])
                        rc = LIB_LLAISYS.llaisysQwen2ModelLoadNamedWeight(
                            self._handle, name.encode("utf-8"),
                            ptr, shape, c_size_t(arr_u16.ndim), lla_dtype
                        )
                        if rc != 0:
                            raise RuntimeError(f"load weight failed (torch fallback): {name}")

    def generate(
        self,
        inputs,
        max_new_tokens=1,
        top_k=1,
        top_p=0.8,
        temperature=0.8,
    ):
        # Greedy only; return the FULL sequence (prompt + new tokens)
        ids = list(inputs)
        for _ in range(max_new_tokens):
            buf = (c_longlong * len(ids))(*ids)
            nxt = LIB_LLAISYS.llaisysQwen2ModelInfer(self._handle, buf, len(ids))
            ids.append(int(nxt))
            # optional early stop:
            # if nxt == self.eos_id: break
        return ids  # <- full sequence

    def __del__(self):
        if getattr(self, "_handle", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._handle)
            self._handle = None