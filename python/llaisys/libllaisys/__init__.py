import os
import sys
import ctypes
from pathlib import Path

# --- 1. 导入基础类型 (无依赖，最先导入) ---
from .llaisys_types import (
    llaisysDeviceType_t,
    DeviceType,
    llaisysDataType_t,
    DataType,
    llaisysMemcpyKind_t,
    MemcpyKind,
    llaisysStream_t,
)

# --- 2. 加载动态库 ---
def load_shared_library():
    # 优先查找当前包目录 (pip install 或 xmake install 后通常在这里)
    lib_dir = Path(__file__).parent
    
    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    lib_path = lib_dir / libname

    # 如果当前目录没有，尝试去项目的 build 目录查找 (开发调试用)
    if not lib_path.exists():
        # 假设结构是 python/llaisys/libllaisys/ -> ... -> build/
        project_root = lib_dir.parent.parent.parent
        # 搜索 build 目录下所有可能的 libllaisys
        candidates = list(project_root.glob(f"build/**/{libname}"))
        if candidates:
            # 优先选 release，没有则选第一个找到的
            release_libs = [p for p in candidates if "release" in str(p)]
            lib_path = release_libs[0] if release_libs else candidates[0]
            print(f"[INFO] libllaisys not found in package, using build artifact: {lib_path}")

    if not lib_path.exists():
        # 最后尝试让系统加载器去 PATH / LD_LIBRARY_PATH 里找
        try:
            return ctypes.CDLL(libname)
        except OSError:
            raise FileNotFoundError(
                f"Shared library '{libname}' not found at {lib_path} and not in system library paths. "
                "Please run 'xmake install' or check your build."
            )

    return ctypes.CDLL(str(lib_path))

# 加载库实例 (这是全局单例)
LIB_LLAISYS = load_shared_library()

# --- 3. 导入子模块定义 ---
# 这些模块定义了 ctypes 类型 (如 llaisysTensor_t) 和加载函数
from .tensor import llaisysTensor_t, load_tensor
from .runtime import LlaisysRuntimeAPI, load_runtime
from .ops import load_ops
# 注意：qwen2 可能依赖前面的类型，所以放在后面导入
from .qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights, load_qwen2_api
from .llama import LlaisysLlamaMeta, LlaisysLlamaWeights, load_llama_api

# --- 4. 执行函数绑定 ---
# 将库句柄传递给各个模块，完成 argtypes/restypes 的设置
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)
load_qwen2_api(LIB_LLAISYS)
load_llama_api(LIB_LLAISYS)

# --- 5. 导出公共符号 (解决 ImportError 的关键) ---
__all__ = [
    "LIB_LLAISYS",
    # Types
    "llaisysTensor_t",
    "llaisysDeviceType_t", "DeviceType",
    "llaisysDataType_t", "DataType",
    "llaisysMemcpyKind_t", "MemcpyKind",
    "llaisysStream_t",
    "LlaisysRuntimeAPI",
    "LlaisysQwen2Meta", "LlaisysQwen2Weights",
    "LlaisysLlamaMeta", "LlaisysLlamaWeights",
]
