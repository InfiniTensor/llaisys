from ctypes import (
    Structure,
    POINTER,
    c_void_p,
    c_size_t,
    c_int,
    c_int64,
)
from ..llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from ..tensor import llaisysTensor_t
from .qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights


# TP Model handle type
llaisysQwen2ModelTP_t = c_void_p


def load_qwen2_tp(lib):
    # llaisysQwen2ModelTPCreate
    lib.llaisysQwen2ModelTPCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),  # meta
        POINTER(c_int),             # device_ids
        c_int,                      # world_size
    ]
    lib.llaisysQwen2ModelTPCreate.restype = llaisysQwen2ModelTP_t

    # llaisysQwen2ModelTPDestroy
    lib.llaisysQwen2ModelTPDestroy.argtypes = [llaisysQwen2ModelTP_t]
    lib.llaisysQwen2ModelTPDestroy.restype = None

    # llaisysQwen2ModelTPWeights
    lib.llaisysQwen2ModelTPWeights.argtypes = [
        llaisysQwen2ModelTP_t,  # model
        c_int,                  # rank
    ]
    lib.llaisysQwen2ModelTPWeights.restype = POINTER(LlaisysQwen2Weights)

    # llaisysQwen2ModelTPInfer
    lib.llaisysQwen2ModelTPInfer.argtypes = [
        llaisysQwen2ModelTP_t,  # model
        POINTER(c_int64),       # token_ids
        c_size_t,               # ntoken
    ]
    lib.llaisysQwen2ModelTPInfer.restype = c_int64

    # llaisysQwen2ModelTPResetCache
    lib.llaisysQwen2ModelTPResetCache.argtypes = [llaisysQwen2ModelTP_t]
    lib.llaisysQwen2ModelTPResetCache.restype = None

    # llaisysQwen2ModelTPGetWorldSize
    lib.llaisysQwen2ModelTPGetWorldSize.argtypes = [llaisysQwen2ModelTP_t]
    lib.llaisysQwen2ModelTPGetWorldSize.restype = c_int
