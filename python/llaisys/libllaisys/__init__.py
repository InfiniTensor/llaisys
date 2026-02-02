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
from .llaisys_types import LlaisysQwen2Meta
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

    return ctypes.CDLL(str(lib_path))

def load_qwen2_api(lib):
    try:
        if hasattr(lib,'llaisysQwen2ModelCreate'):
            lib.llaisysQwen2ModelCreate.argtypes=[
                ctypes.POINTER(LlaisysQwen2Meta),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int
            ]
            lib.llaisysQwen2ModelCreate.restype=ctypes.c_void_p
        if hasattr(lib,'llaisysQwen2LoadWeight'):
            lib.llaisysQwen2LoadWeight.argtypes=[
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_int
            ]
            lib.llaisysQwen2LoadWeight.restype=None
    except Exception as e:
       print(f"Warning: Failed to load Qwen2 API signatures. {e}")
def llaisys_qwen2_create(meta,device_id):
    return LIB_LLAISYS.llaisysQwen2ModelCreate(
        ctypes.byref(meta),
        device_id,
        None,
        0
    )
def llaisys_qwen2_load_weight(model_handle,name,data_ptr,shape,ndim,dtype):
    LIB_LLAISYS.llaisysQwen2LoadWeight(
        model_handle,
        name,
        data_ptr,
        shape,
        ndim,
        dtype
    )

LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)
load_qwen2_api(LIB_LLAISYS)


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
    "LlaisysQwen2Meta",
    "llaisys_qwen2_create",
    "llaisys_qwen2_load_weight"
]
