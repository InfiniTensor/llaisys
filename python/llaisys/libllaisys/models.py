import ctypes
from ctypes import c_size_t, c_int, c_int64, c_float, POINTER

from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t


class LlaisysQwen2Meta(ctypes.Structure):
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


class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]


def load_models(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        ctypes.POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        ctypes.POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = ctypes.c_void_p

    lib.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]
    lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelGetWeight.argtypes = [ctypes.c_void_p, c_int, c_size_t]
    lib.llaisysQwen2ModelGetWeight.restype = llaisysTensor_t

    lib.llaisysQwen2ModelInfer.argtypes = [ctypes.c_void_p, ctypes.POINTER(c_int64), c_size_t, c_int, c_float, c_float]
    lib.llaisysQwen2ModelInfer.restype = c_int64


__all__ = [
    "LlaisysQwen2Meta",
    "LlaisysQwen2Weights",
    "load_models",
]
