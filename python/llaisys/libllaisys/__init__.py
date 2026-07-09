import os
import sys
import ctypes
from pathlib import Path

from .runtime import load_runtime
from .runtime import LlaisysRuntimeAPI
from .llaisys_types import llaisysDeviceType_t, DeviceType
from .llaisys_types import llaisysDataType_t, DataType
from .llaisys_types import llaisysMemcpyKind_t, MemcpyKind
from .llaisys_types import llaisysStream_t
from .tensor import llaisysTensor_t
from .tensor import load_tensor
from .ops import load_ops


def load_shared_library():
    lib_dir = Path(__file__).parent

    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}")
    # 👇👇👇 在 return 之前，加上这两行终极黑魔法 👇👇👇
    import ctypes
    # 告诉操作系统：提前把 OpenBLAS 加载到全局内存里！
    ctypes.CDLL("libopenblas.so.0", mode=ctypes.RTLD_GLOBAL)
    
    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)

# ============================================================================
# Qwen2 Bindings
# ============================================================================

class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int),
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

try:
    LIB_LLAISYS.llaisysQwen2ModelCreate.restype = ctypes.c_void_p
    LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [ctypes.POINTER(LlaisysQwen2Meta), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

    LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None
    LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]

    LIB_LLAISYS.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)
    LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]

    LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64
    LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]

    if hasattr(LIB_LLAISYS, 'llaisysTensorData'):
        LIB_LLAISYS.llaisysTensorData.restype = ctypes.c_void_p
        LIB_LLAISYS.llaisysTensorData.argtypes = [ctypes.c_void_p]
except AttributeError:
    pass 

__all__ = [
    "LIB_LLAISYS",
    "LlaisysRuntimeAPI",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "LlaisysQwen2Meta",
    "LlaisysQwen2Weights"
]