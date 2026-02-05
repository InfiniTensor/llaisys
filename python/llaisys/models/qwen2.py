from __future__ import annotations
from typing import Sequence, Optional
from pathlib import Path
import ctypes
import numpy as np
import safetensors
import json

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType

llaisysTensor_t = ctypes.c_void_p


class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int),  # llaisysDataType_t
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]


class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_o_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_gate_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_up_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_down_w", ctypes.POINTER(llaisysTensor_t)),
    ]


def _dtype_is_bf16(dt) -> bool:
    return int(dt) == int(DataType.BF16)


def _dtype_is_f16(dt) -> bool:
    return int(dt) == int(DataType.F16)


def _dtype_is_f32(dt) -> bool:
    return int(dt) == int(DataType.F32)


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.lib = LIB_LLAISYS
        self.device = device

        model_path = Path(model_path)
        cfg_path = model_path / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json in {model_path}")

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        # ---- model meta ----
        nlayer = int(cfg["num_hidden_layers"])
        hs = int(cfg["hidden_size"])
        nh = int(cfg["num_attention_heads"])
        nkvh = int(cfg.get("num_key_value_heads", nh))
        di = int(cfg["intermediate_size"])
        maxseq = int(cfg.get("max_position_embeddings", 32768))
        voc = int(cfg["vocab_size"])
        dh = hs // nh

        eps = float(cfg.get("rms_norm_eps", 1e-6))
        theta = float(cfg.get("rope_theta", 10000.0))

        eos = cfg.get("eos_token_id", -1)
        if isinstance(eos, list):
            end_token = int(eos[0])
        else:
            end_token = int(eos)

        # assignment uses bf16
        dtype = DataType.BF16

        self.meta = LlaisysQwen2Meta(
            dtype=int(dtype),
            nlayer=nlayer,
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=dh,
            di=di,
            maxseq=maxseq,
            voc=voc,
            epsilon=eps,
            theta=theta,
            end_token=end_token,
        )

        # ---- bind APIs ----
        self.lib.llaisysQwen2ModelCreate.argtypes = [
            ctypes.POINTER(LlaisysQwen2Meta),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.llaisysQwen2ModelCreate.restype = ctypes.c_void_p

        self.lib.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]
        self.lib.llaisysQwen2ModelDestroy.restype = None

        self.lib.llaisysQwen2ModelReset.argtypes = [ctypes.c_void_p]
        self.lib.llaisysQwen2ModelReset.restype = None

        self.lib.llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]
        self.lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

        self.lib.llaisysQwen2ModelInfer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
        ]
        self.lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64

        # tensorLoad already correct in your bindings
        self.lib.tensorLoad.argtypes = [llaisysTensor_t, ctypes.c_void_p]
        self.lib.tensorLoad.restype = None

        # tensorGetData helper (for tied lm_head fallback)
        self.lib.tensorGetData.argtypes = [llaisysTensor_t]
        self.lib.tensorGetData.restype = ctypes.c_void_p

        # ---- create backend ----
        self.model = self.lib.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            int(device),
            None,
            0,
        )
        if not self.model:
            raise RuntimeError("Failed to create backend Qwen2 model")

        self.weights = self.lib.llaisysQwen2ModelWeights(self.model).contents

        # ---- weight loader ----
        def to_backend_bits(arr: np.ndarray) -> np.ndarray:
            """
            Return a contiguous numpy array whose raw bytes match backend tensor dtype.
            For BF16/F16 backend, we pass uint16 bits.
            """
            if _dtype_is_bf16(dtype):
                # Typical cases when framework="numpy":
                # - np.uint16 already (bf16 bits)
                # - np.float16/np.float32 (fallback)
                if arr.dtype == np.uint16:
                    return np.ascontiguousarray(arr)
                if arr.dtype == np.float32:
                    x = arr.astype(np.float32, copy=False)
                    bits = x.view(np.uint32)
                    rb = (0x7FFF + ((bits >> 16) & 1)).astype(np.uint32)
                    bf16 = ((bits + rb) >> 16).astype(np.uint16)
                    return np.ascontiguousarray(bf16)
                if arr.dtype == np.float16:
                    x = arr.astype(np.float32)
                    bits = x.view(np.uint32)
                    rb = (0x7FFF + ((bits >> 16) & 1)).astype(np.uint32)
                    bf16 = ((bits + rb) >> 16).astype(np.uint16)
                    return np.ascontiguousarray(bf16)
                # Some environments may expose bfloat16 dtype as a string
                if str(arr.dtype) == "bfloat16":
                    return np.ascontiguousarray(arr.view(np.uint16))
                raise TypeError(f"Unsupported weight dtype for BF16 backend: {arr.dtype}")

            if _dtype_is_f16(dtype):
                if arr.dtype == np.uint16:
                    return np.ascontiguousarray(arr)
                if arr.dtype == np.float16:
                    return np.ascontiguousarray(arr.view(np.uint16))
                if arr.dtype == np.float32:
                    return np.ascontiguousarray(arr.astype(np.float16).view(np.uint16))
                raise TypeError(f"Unsupported weight dtype for F16 backend: {arr.dtype}")

            # F32
            return np.ascontiguousarray(arr.astype(np.float32, copy=False))

        def load(handle: llaisysTensor_t, arr: np.ndarray):
            a = to_backend_bits(arr)
            self.lib.tensorLoad(handle, a.ctypes.data_as(ctypes.c_void_p))

        def normalize_name(name: str) -> str:
            return ("model." + name[len("model.model.") :]) if name.startswith("model.model.") else name

        lm_head_loaded = False

        st_files = sorted(model_path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No *.safetensors found in {model_path}")

        # use framework="numpy" to remove torch dependency ----
        for file in st_files:
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")

            for raw_name in data_.keys():
                name = normalize_name(raw_name)
                arr = data_.get_tensor(raw_name)  # numpy.ndarray

                # embeddings / final norm / lm head
                if name == "model.embed_tokens.weight":
                    load(self.weights.in_embed, arr)
                    continue
                if name in ("lm_head.weight", "model.lm_head.weight"):
                    load(self.weights.out_embed, arr)
                    lm_head_loaded = True
                    continue
                if name == "model.norm.weight":
                    load(self.weights.out_norm_w, arr)
                    continue

                if not name.startswith("model.layers."):
                    continue

                parts = name.split(".")
                if len(parts) < 4:
                    continue
                try:
                    li = int(parts[2])
                except Exception:
                    continue

                suffix = ".".join(parts[3:])

                if suffix == "input_layernorm.weight":
                    load(self.weights.attn_norm_w[li], arr)

                elif suffix == "self_attn.q_proj.weight":
                    load(self.weights.attn_q_w[li], arr)
                elif suffix == "self_attn.q_proj.bias":
                    load(self.weights.attn_q_b[li], arr)

                elif suffix == "self_attn.k_proj.weight":
                    load(self.weights.attn_k_w[li], arr)
                elif suffix == "self_attn.k_proj.bias":
                    load(self.weights.attn_k_b[li], arr)

                elif suffix == "self_attn.v_proj.weight":
                    load(self.weights.attn_v_w[li], arr)
                elif suffix == "self_attn.v_proj.bias":
                    load(self.weights.attn_v_b[li], arr)

                elif suffix == "self_attn.o_proj.weight":
                    load(self.weights.attn_o_w[li], arr)

                elif suffix == "post_attention_layernorm.weight":
                    load(self.weights.mlp_norm_w[li], arr)

                elif suffix == "mlp.gate_proj.weight":
                    load(self.weights.mlp_gate_w[li], arr)
                elif suffix == "mlp.up_proj.weight":
                    load(self.weights.mlp_up_w[li], arr)
                elif suffix == "mlp.down_proj.weight":
                    load(self.weights.mlp_down_w[li], arr)

        # tied lm_head fallback: if missing, copy embed -> out_embed
        if not lm_head_loaded:
            out_ptr = self.lib.tensorGetData(self.weights.out_embed)
            in_ptr = self.lib.tensorGetData(self.weights.in_embed)
            if out_ptr and in_ptr:
                if _dtype_is_bf16(dtype) or _dtype_is_f16(dtype):
                    nbytes = voc * hs * 2
                elif _dtype_is_f32(dtype):
                    nbytes = voc * hs * 4
                else:
                    raise RuntimeError("Unsupported dtype for tied lm_head copy")
                ctypes.memmove(out_ptr, in_ptr, nbytes)

    def __del__(self):
        m = getattr(self, "model", None)
        if m:
            try:
                self.lib.llaisysQwen2ModelDestroy(m)
            except Exception:
                pass
            self.model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        # Greedy for test: top_k=1, top_p=1.0, temperature=1.0
        # We ignore sampling params intentionally.
        tokens = [int(x) for x in inputs]
        eos = int(self.meta.end_token)

        # Start a new request: reset KV-cache cursor/state in backend.
        # This makes repeated generate() calls on the same model safe and deterministic.
        self.lib.llaisysQwen2ModelReset(self.model)

        arr = (ctypes.c_int64 * len(tokens))(*tokens)
        nxt = int(self.lib.llaisysQwen2ModelInfer(self.model, arr, len(tokens)))
        tokens.append(nxt)
        if nxt == eos:
            return tokens

        for _ in range(max_new_tokens - 1):
            last = tokens[-1]
            arr1 = (ctypes.c_int64 * 1)(last)
            nxt = int(self.lib.llaisysQwen2ModelInfer(self.model, arr1, 1))
            tokens.append(nxt)
            if nxt == eos:
                break

        return tokens

# from __future__ import annotations
# from typing import Sequence, Optional
# from pathlib import Path
# import ctypes
# import numpy as np
# import safetensors
# import json

# from ..libllaisys import LIB_LLAISYS
# from ..libllaisys import DeviceType, DataType

# import torch


# llaisysTensor_t = ctypes.c_void_p


# class LlaisysQwen2Meta(ctypes.Structure):
#     _fields_ = [
#         ("dtype", ctypes.c_int),  # llaisysDataType_t
#         ("nlayer", ctypes.c_size_t),
#         ("hs", ctypes.c_size_t),
#         ("nh", ctypes.c_size_t),
#         ("nkvh", ctypes.c_size_t),
#         ("dh", ctypes.c_size_t),
#         ("di", ctypes.c_size_t),
#         ("maxseq", ctypes.c_size_t),
#         ("voc", ctypes.c_size_t),
#         ("epsilon", ctypes.c_float),
#         ("theta", ctypes.c_float),
#         ("end_token", ctypes.c_int64),
#     ]


# class LlaisysQwen2Weights(ctypes.Structure):
#     _fields_ = [
#         ("in_embed", llaisysTensor_t),
#         ("out_embed", llaisysTensor_t),
#         ("out_norm_w", llaisysTensor_t),
#         ("attn_norm_w", ctypes.POINTER(llaisysTensor_t)),
#         ("attn_q_w", ctypes.POINTER(llaisysTensor_t)),
#         ("attn_q_b", ctypes.POINTER(llaisysTensor_t)),
#         ("attn_k_w", ctypes.POINTER(llaisysTensor_t)),
#         ("attn_k_b", ctypes.POINTER(llaisysTensor_t)),
#         ("attn_v_w", ctypes.POINTER(llaisysTensor_t)),
#         ("attn_v_b", ctypes.POINTER(llaisysTensor_t)),
#         ("attn_o_w", ctypes.POINTER(llaisysTensor_t)),
#         ("mlp_norm_w", ctypes.POINTER(llaisysTensor_t)),
#         ("mlp_gate_w", ctypes.POINTER(llaisysTensor_t)),
#         ("mlp_up_w", ctypes.POINTER(llaisysTensor_t)),
#         ("mlp_down_w", ctypes.POINTER(llaisysTensor_t)),
#     ]


# def _dtype_is_bf16(dt) -> bool:
#     return int(dt) == int(DataType.BF16)


# def _dtype_is_f16(dt) -> bool:
#     return int(dt) == int(DataType.F16)


# def _dtype_is_f32(dt) -> bool:
#     return int(dt) == int(DataType.F32)


# class Qwen2:
#     def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
#         #self.lib = LIB_LLAISYS.lib
#         self.lib = LIB_LLAISYS

#         self.device = device

#         model_path = Path(model_path)
#         cfg_path = model_path / "config.json"
#         if not cfg_path.exists():
#             raise FileNotFoundError(f"Missing config.json in {model_path}")

#         cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

#         # ---- model meta ----
#         nlayer = int(cfg["num_hidden_layers"])
#         hs = int(cfg["hidden_size"])
#         nh = int(cfg["num_attention_heads"])
#         nkvh = int(cfg.get("num_key_value_heads", nh))
#         di = int(cfg["intermediate_size"])
#         maxseq = int(cfg.get("max_position_embeddings", 32768))
#         voc = int(cfg["vocab_size"])
#         dh = hs // nh

#         eps = float(cfg.get("rms_norm_eps", 1e-6))
#         theta = float(cfg.get("rope_theta", 10000.0))

#         # eos token
#         eos = cfg.get("eos_token_id", -1)
#         if isinstance(eos, list):
#             end_token = int(eos[0])
#         else:
#             end_token = int(eos)

#         # assignment uses bf16
#         dtype = DataType.BF16

#         self.meta = LlaisysQwen2Meta(
#             dtype=int(dtype),
#             nlayer=nlayer,
#             hs=hs,
#             nh=nh,
#             nkvh=nkvh,
#             dh=dh,
#             di=di,
#             maxseq=maxseq,
#             voc=voc,
#             epsilon=eps,
#             theta=theta,
#             end_token=end_token,
#         )

#         # ---- bind APIs ----
#         self.lib.llaisysQwen2ModelCreate.argtypes = [
#             ctypes.POINTER(LlaisysQwen2Meta),
#             ctypes.c_int,
#             ctypes.POINTER(ctypes.c_int),
#             ctypes.c_int,
#         ]
#         self.lib.llaisysQwen2ModelCreate.restype = ctypes.c_void_p

#         self.lib.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]
#         self.lib.llaisysQwen2ModelDestroy.restype = None

#         self.lib.llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]
#         self.lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

#         self.lib.llaisysQwen2ModelInfer.argtypes = [
#             ctypes.c_void_p,
#             ctypes.POINTER(ctypes.c_int64),
#             ctypes.c_size_t,
#         ]
#         self.lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64

#         # tensorLoad already correct in your bindings
#         self.lib.tensorLoad.argtypes = [llaisysTensor_t, ctypes.c_void_p]
#         self.lib.tensorLoad.restype = None

#         # tensorGetData helper (for tied lm_head fallback)
#         self.lib.tensorGetData.argtypes = [llaisysTensor_t]
#         self.lib.tensorGetData.restype = ctypes.c_void_p

#         # ---- create backend ----
#         self.model = self.lib.llaisysQwen2ModelCreate(
#             ctypes.byref(self.meta),
#             int(device),
#             None,
#             0,
#         )
#         if not self.model:
#             raise RuntimeError("Failed to create backend Qwen2 model")

#         self.weights = self.lib.llaisysQwen2ModelWeights(self.model).contents

#         # ---- weight loader ----
#         def to_backend_bits(arr: np.ndarray) -> np.ndarray:
#             """
#             Return a contiguous numpy array whose raw bytes match backend tensor dtype.
#             For BF16/F16 backend, we pass uint16 bits.
#             """
#             if _dtype_is_bf16(dtype):
#                 # safetensors may produce:
#                 # - numpy uint16 already (bf16 bits)
#                 # - numpy bfloat16 (rare)
#                 # - float32 (shouldn't for model weights, but keep robust)
#                 if arr.dtype == np.uint16:
#                     return np.ascontiguousarray(arr)
#                 if str(arr.dtype) == "bfloat16":
#                     return np.ascontiguousarray(arr.view(np.uint16))
#                 if arr.dtype == np.float32:
#                     x = arr.astype(np.float32, copy=False)
#                     bits = x.view(np.uint32)
#                     rb = (0x7FFF + ((bits >> 16) & 1)).astype(np.uint32)
#                     bf16 = ((bits + rb) >> 16).astype(np.uint16)
#                     return np.ascontiguousarray(bf16)
#                 # float16 -> float32 -> bf16
#                 if arr.dtype == np.float16:
#                     x = arr.astype(np.float32)
#                     bits = x.view(np.uint32)
#                     rb = (0x7FFF + ((bits >> 16) & 1)).astype(np.uint32)
#                     bf16 = ((bits + rb) >> 16).astype(np.uint16)
#                     return np.ascontiguousarray(bf16)
#                 raise TypeError(f"Unsupported weight dtype for BF16 backend: {arr.dtype}")

#             if _dtype_is_f16(dtype):
#                 if arr.dtype == np.uint16:
#                     return np.ascontiguousarray(arr)
#                 if arr.dtype == np.float16:
#                     return np.ascontiguousarray(arr.view(np.uint16))
#                 if arr.dtype == np.float32:
#                     return np.ascontiguousarray(arr.astype(np.float16).view(np.uint16))
#                 raise TypeError(f"Unsupported weight dtype for F16 backend: {arr.dtype}")

#             # F32
#             return np.ascontiguousarray(arr.astype(np.float32, copy=False))

#         def load(handle: llaisysTensor_t, arr: np.ndarray):
#             a = to_backend_bits(arr)
#             self.lib.tensorLoad(handle, a.ctypes.data_as(ctypes.c_void_p))

#         # For robustness: accept both "model.*" and "model.model.*"
#         # def normalize_name(name: str) -> str:
#         #     if name.startswith("model.model."):
#         #         return "model." + name[len("model.model.") :]
#         #     return name
#         def normalize_name(name: str) -> str:
#             return ("model." + name[len("model.model.") :]) if name.startswith("model.model.") else name

#         # flags to detect if lm_head loaded
#         lm_head_loaded = False

#         st_files = sorted(model_path.glob("*.safetensors"))
#         if not st_files:
#             raise FileNotFoundError(f"No *.safetensors found in {model_path}")

#         for file in st_files:
#             #data_ = safetensors.safe_open(file, framework="numpy", device="cpu")

#             data_ = safetensors.safe_open(file, framework="pt", device="cpu")

#             for raw_name in data_.keys():
#                 name = normalize_name(raw_name)
#                 # arr = data_.get_tensor(raw_name)
#                 t = data_.get_tensor(raw_name)  # torch.Tensor
#                 # 统一转成 uint16 bits（BF16/F16 都是 2 bytes）
#                 if t.dtype == getattr(__import__("torch"), "bfloat16"):
#                     arr = t.contiguous().view(__import__("torch").uint16).cpu().numpy()
#                 elif t.dtype == getattr(__import__("torch"), "float16"):
#                     arr = t.contiguous().view(__import__("torch").uint16).cpu().numpy()
#                 elif t.dtype == getattr(__import__("torch"), "float32"):
#                     arr = t.contiguous().cpu().numpy()
#                 else:
#                     arr = t.contiguous().cpu().numpy()

#                 # embeddings / final norm / lm head
#                 if name == "model.embed_tokens.weight":
#                     load(self.weights.in_embed, arr)
#                     continue
#                 if name in ("lm_head.weight", "model.lm_head.weight"):
#                     load(self.weights.out_embed, arr)
#                     lm_head_loaded = True
#                     continue
#                 if name == "model.norm.weight":
#                     load(self.weights.out_norm_w, arr)
#                     continue

#                 if not name.startswith("model.layers."):
#                     continue

#                 parts = name.split(".")
#                 if len(parts) < 4:
#                     continue
#                 try:
#                     li = int(parts[2])
#                 except Exception:
#                     continue

#                 suffix = ".".join(parts[3:])

#                 if suffix == "input_layernorm.weight":
#                     load(self.weights.attn_norm_w[li], arr)

#                 elif suffix == "self_attn.q_proj.weight":
#                     load(self.weights.attn_q_w[li], arr)
#                 elif suffix == "self_attn.q_proj.bias":
#                     load(self.weights.attn_q_b[li], arr)

#                 elif suffix == "self_attn.k_proj.weight":
#                     load(self.weights.attn_k_w[li], arr)
#                 elif suffix == "self_attn.k_proj.bias":
#                     load(self.weights.attn_k_b[li], arr)

#                 elif suffix == "self_attn.v_proj.weight":
#                     load(self.weights.attn_v_w[li], arr)
#                 elif suffix == "self_attn.v_proj.bias":
#                     load(self.weights.attn_v_b[li], arr)

#                 elif suffix == "self_attn.o_proj.weight":
#                     load(self.weights.attn_o_w[li], arr)

#                 elif suffix == "post_attention_layernorm.weight":
#                     load(self.weights.mlp_norm_w[li], arr)

#                 elif suffix == "mlp.gate_proj.weight":
#                     load(self.weights.mlp_gate_w[li], arr)
#                 elif suffix == "mlp.up_proj.weight":
#                     load(self.weights.mlp_up_w[li], arr)
#                 elif suffix == "mlp.down_proj.weight":
#                     load(self.weights.mlp_down_w[li], arr)

#         # tied lm_head fallback: if missing, copy embed -> out_embed
#         if not lm_head_loaded:
#             out_ptr = self.lib.tensorGetData(self.weights.out_embed)
#             in_ptr = self.lib.tensorGetData(self.weights.in_embed)
#             if out_ptr and in_ptr:
#                 # copy bytes
#                 if _dtype_is_bf16(dtype) or _dtype_is_f16(dtype):
#                     nbytes = voc * hs * 2
#                 elif _dtype_is_f32(dtype):
#                     nbytes = voc * hs * 4
#                 else:
#                     raise RuntimeError("Unsupported dtype for tied lm_head copy")
#                 ctypes.memmove(out_ptr, in_ptr, nbytes)

#     def __del__(self):
#         m = getattr(self, "model", None)
#         if m:
#             try:
#                 self.lib.llaisysQwen2ModelDestroy(m)
#             except Exception:
#                 pass
#             self.model = None

#     def generate(
#         self,
#         inputs: Sequence[int],
#         max_new_tokens: Optional[int] = None,
#         top_k: int = 1,
#         top_p: float = 0.8,
#         temperature: float = 0.8,
#     ):
#         if max_new_tokens is None:
#             max_new_tokens = 128

#         # Greedy for test: top_k=1, top_p=1.0, temperature=1.0
#         # We ignore sampling params intentionally.
#         tokens = [int(x) for x in inputs]
#         eos = int(self.meta.end_token)

#         # # prefill once
#         # arr = (ctypes.c_int64 * len(tokens))(*tokens)
#         # nxt = int(self.lib.llaisysQwen2ModelInfer(self.model, arr, len(tokens)))
#         # tokens.append(nxt)

#         # # decode
#         # for _ in range(max_new_tokens - 1):
#         #     last = tokens[-1]
#         #     arr1 = (ctypes.c_int64 * 1)(last)
#         #     nxt = int(self.lib.llaisysQwen2ModelInfer(self.model, arr1, 1))
#         #     tokens.append(nxt)
#         # 先把 prompt 一次性喂进去，拿到第一个 next token
#         arr = (ctypes.c_int64 * len(tokens))(*tokens)
#         nxt = int(self.lib.llaisysQwen2ModelInfer(self.model, arr, len(tokens)))
#         tokens.append(nxt)
#         if nxt == eos:
#             return tokens

#         # 后续增量生成：遇到 eos 立刻停止（对齐 HF generate）
#         for _ in range(max_new_tokens - 1):
#             last = tokens[-1]
#             arr1 = (ctypes.c_int64 * 1)(last)
#             nxt = int(self.lib.llaisysQwen2ModelInfer(self.model, arr1, 1))
#             tokens.append(nxt)
#             if nxt == eos:
#                 break

#         return tokens
