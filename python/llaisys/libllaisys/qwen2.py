from ctypes import (
    Structure,
    POINTER,
    c_int,
    c_size_t,
    c_float,
    c_int64,
    c_void_p,
    c_char_p,
    c_bool,
)
import sys

from .llaisys_types import llaisysDeviceType_t, llaisysDataType_t
from .tensor import llaisysTensor_t


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


def load_qwen2_model(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = c_void_p

    lib.llaisysQwen2SetWeights.argtypes = [c_void_p, c_int, c_int, llaisysTensor_t]
    lib.llaisysQwen2SetWeights.restype = None

    lib.llaisysQwen2ModelInfer.argtypes = [
        c_void_p,
        POINTER(c_int64),
        POINTER(c_int64),
        c_size_t,
        c_bool,
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelInferSample.argtypes = [
        c_void_p,
        POINTER(c_int64),
        POINTER(c_int64),
        c_size_t,
        c_bool,
        c_int,
        c_float,
        c_float,
    ]
    lib.llaisysQwen2ModelInferSample.restype = c_int64

    lib.llaisysQwen2ModelWeights.argtypes = [c_void_p]
    lib.llaisysQwen2ModelWeights.restype = c_void_p

    lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    return lib


__all__ = ["LlaisysQwen2Meta", "load_qwen2_model"]
