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
        # 尝试开发环境构建路径
        build_path = os.path.join(lib_dir, "..", "..", "..", "build", "windows", "x64", "release", "llaisys.dll")
        if os.path.isfile(build_path):
             return ctypes.CDLL(str(build_path))
        raise FileNotFoundError(f"Shared library not found: {lib_path}")

    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)

# =========================================================================
#  Task 3: Qwen2 C API Registration
# =========================================================================

# 1. 定义 Meta 结构体 (对应 qwen2.h 中的 LlaisysQwen2Meta)
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int),          # llaisysDataType_t
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

# 2. 定义 Weights 结构体 (对应 qwen2.h 中的 LlaisysQwen2Weights)
# 注意：C++ 中的 llaisysTensor_t 是指针，这里对应 ctypes.c_void_p
class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", ctypes.c_void_p),    
        ("out_embed", ctypes.c_void_p),
        ("out_norm_w", ctypes.c_void_p),
        ("attn_norm_w", ctypes.POINTER(ctypes.c_void_p)), # 指针数组
        ("attn_q_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_q_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_k_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_k_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_v_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_v_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_o_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_norm_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_gate_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_up_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_down_w", ctypes.POINTER(ctypes.c_void_p)),
    ]

# 3. 注册函数签名
if hasattr(LIB_LLAISYS, "llaisysQwen2ModelCreate"):
    LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [
        ctypes.POINTER(LlaisysQwen2Meta),
        ctypes.c_int, 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.c_int
    ]
    LIB_LLAISYS.llaisysQwen2ModelCreate.restype = ctypes.c_void_p # Model*

if hasattr(LIB_LLAISYS, "llaisysQwen2ModelDestroy"):
    LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]
    LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

if hasattr(LIB_LLAISYS, "llaisysQwen2ModelWeights"):
    LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]
    LIB_LLAISYS.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

if hasattr(LIB_LLAISYS, "llaisysQwen2ModelInfer"):
    LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t
    ]
    LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64

__all__ = [
    "LIB_LLAISYS",
    "LlaisysRuntimeAPI",
    "llaisysStream_t",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
    "LlaisysQwen2Meta",    # 新增
    "LlaisysQwen2Weights"  # 新增
]