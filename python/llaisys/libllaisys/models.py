from ctypes import (
    Structure,
    POINTER,
    c_size_t,
    c_int,
    c_float,
    c_int64,
    c_uint32,
    c_void_p,
)

from .llaisys_types import llaisysDeviceType_t, llaisysDataType_t
from .tensor import llaisysTensor_t


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype",      llaisysDataType_t),
        ("nlayer",     c_size_t),
        ("hs",         c_size_t),
        ("nh",         c_size_t),
        ("nkvh",       c_size_t),
        ("dh",         c_size_t),
        ("di",         c_size_t),
        ("maxseq",     c_size_t),
        ("voc",        c_size_t),
        ("epsilon",    c_float),
        ("theta",      c_float),
        ("end_token",  c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed",       llaisysTensor_t),
        ("out_embed",      llaisysTensor_t),
        ("out_norm_w",     llaisysTensor_t),
        ("attn_norm_w",    POINTER(llaisysTensor_t)),
        ("attn_q_w",       POINTER(llaisysTensor_t)),
        ("attn_q_b",       POINTER(llaisysTensor_t)),
        ("attn_k_w",       POINTER(llaisysTensor_t)),
        ("attn_k_b",       POINTER(llaisysTensor_t)),
        ("attn_v_w",       POINTER(llaisysTensor_t)),
        ("attn_v_b",       POINTER(llaisysTensor_t)),
        ("attn_o_w",       POINTER(llaisysTensor_t)),
        ("mlp_norm_w",     POINTER(llaisysTensor_t)),
        ("mlp_gate_w",     POINTER(llaisysTensor_t)),
        ("mlp_up_w",       POINTER(llaisysTensor_t)),
        ("mlp_down_w",     POINTER(llaisysTensor_t)),
    ]


class LlaisysSamplingParams(Structure):
    _fields_ = [
        ("top_k",        c_int),
        ("top_p",        c_float),
        ("temperature",  c_float),
        ("seed",         c_uint32),
    ]


LlaisysQwen2Model = c_void_p


def load_models(lib):
    # Model lifecycle
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model

    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelDestroy.restype = None

    # Weight access
    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    # Inference interfaces
    lib.llaisysQwen2ModelInfer.argtypes = [LlaisysQwen2Model, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelPrefill.argtypes = [LlaisysQwen2Model, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelPrefill.restype = c_int64

    lib.llaisysQwen2ModelStep.argtypes = [LlaisysQwen2Model, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelStep.restype = c_int64

    # Sampling variants
    lib.llaisysQwen2ModelInferSampling.argtypes = [
        LlaisysQwen2Model,
        POINTER(c_int64),
        c_size_t,
        POINTER(LlaisysSamplingParams),
    ]
    lib.llaisysQwen2ModelInferSampling.restype = c_int64

    lib.llaisysQwen2ModelInferSamplingEx.argtypes = [
        LlaisysQwen2Model,
        POINTER(c_int64),
        c_size_t,
        c_int,
        c_float,
        c_float,
        c_uint32,
    ]
    lib.llaisysQwen2ModelInferSamplingEx.restype = c_int64

    # KV cache control
    lib.llaisysQwen2ModelResetKVCache.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelResetKVCache.restype = None

    lib.llaisysQwen2ModelSetKVCacheEnabled.argtypes = [LlaisysQwen2Model, c_int]
    lib.llaisysQwen2ModelSetKVCacheEnabled.restype = None


__all__ = [
    "LlaisysQwen2Meta",
    "LlaisysQwen2Weights",
    "LlaisysSamplingParams",
    "LlaisysQwen2Model",
    "load_models",
]