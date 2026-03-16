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
from .qwen2 import load_qwen2
from .qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights, LlaisysQwen2Model_t
from . import nccl_comm


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

    # 预加载 OpenMP 运行时，避免 libllaisys.so 出现 undefined symbol: omp_get_thread_num
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # 若系统无 libgomp 或已链接进 .so，忽略

    lib_path_abs = os.path.abspath(lib_path)
    if os.environ.get("LLAISYS_DEBUG"):
        print(f"[LLAISYS] Loading shared library: {lib_path_abs}")
    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)
load_qwen2(LIB_LLAISYS)
nccl_comm.load_nccl(LIB_LLAISYS)

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
    "LlaisysQwen2Weights",
    "LlaisysQwen2Model_t",
]
