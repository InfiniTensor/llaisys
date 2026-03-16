"""ctypes 绑定：NCCL 通信（项目#5 张量并行）。仅当编译时启用 ENABLE_NCCL 时符号存在。"""
from ctypes import c_int, c_void_p, c_size_t, c_char_p, create_string_buffer
import os

# 与 include/llaisys/nccl_comm.h 一致
LLAISYS_NCCL_UNIQUE_ID_BYTES = 128


def load_nccl(lib):
    """为 lib 绑定 NCCL 相关符号；若未编译 NCCL 则部分可能缺失。"""
    try:
        lib.llaisysNcclGetUniqueId.argtypes = [c_void_p]
        lib.llaisysNcclGetUniqueId.restype = None

        lib.llaisysNcclInitRank.argtypes = [c_int, c_int, c_void_p]
        lib.llaisysNcclInitRank.restype = c_int

        lib.llaisysNcclAllReduce.argtypes = [
            c_void_p, c_void_p, c_size_t,
            c_int,  # llaisysDataType_t
            c_void_p,  # stream
        ]
        lib.llaisysNcclAllReduce.restype = c_int

        lib.llaisysNcclAllGather.argtypes = [
            c_void_p, c_void_p, c_size_t,
            c_int,
            c_void_p,
        ]
        lib.llaisysNcclAllGather.restype = c_int

        lib.llaisysNcclDestroy.argtypes = []
        lib.llaisysNcclDestroy.restype = None

        lib.llaisysNcclGetLastError.argtypes = []
        lib.llaisysNcclGetLastError.restype = c_char_p
        return True
    except AttributeError:
        return False


def get_unique_id(lib):
    """在 rank 0 上调用，返回 LLAISYS_NCCL_UNIQUE_ID_BYTES 字节的 bytes，供广播给其他 rank。"""
    buf = create_string_buffer(LLAISYS_NCCL_UNIQUE_ID_BYTES)
    lib.llaisysNcclGetUniqueId(buf)
    return bytes(buf.raw)


def init_rank(lib, rank: int, world_size: int, unique_id: bytes) -> int:
    """每个进程调用一次；unique_id 来自 rank 0 的 get_unique_id()。返回 0 成功，-1 失败。"""
    if len(unique_id) < LLAISYS_NCCL_UNIQUE_ID_BYTES:
        return -1
    # Python 3 中 create_string_buffer(n).raw 为不可变 bytes，不能切片赋值；用内容初始化 buffer
    data = (unique_id[:LLAISYS_NCCL_UNIQUE_ID_BYTES]).ljust(LLAISYS_NCCL_UNIQUE_ID_BYTES, b"\x00")
    buf = create_string_buffer(data)
    return lib.llaisysNcclInitRank(rank, world_size, buf)


def get_last_error(lib):
    """返回 C 端记录的最近一次 NCCL/CUDA 错误（调试用）。"""
    try:
        p = lib.llaisysNcclGetLastError()
        return p.decode("utf-8") if p else ""
    except AttributeError:
        return ""


def destroy(lib):
    """进程退出前释放 NCCL 通信器。"""
    try:
        lib.llaisysNcclDestroy()
    except AttributeError:
        pass
