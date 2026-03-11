from ctypes import (
    Structure,
    POINTER,
    c_void_p,
    c_int,
    c_int64,
    c_uint64,
    c_size_t,
    c_float,
)

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


class LlaisysQwen2Weights(Structure):
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


def load_qwen2_api(lib):
    """
    Bind C APIs declared in include/llaisys/models/qwen2.h
    """
    # struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta*, device, int* ids, int n)
    lib.llaisysQwen2ModelCreate.restype = c_void_p
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]

    # void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model*)
    lib.llaisysQwen2ModelDestroy.restype = None
    lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]

    # struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model*)
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)
    lib.llaisysQwen2ModelWeights.argtypes = [c_void_p]

    lib.llaisysQwen2ModelReset.restype = None
    lib.llaisysQwen2ModelReset.argtypes = [c_void_p]

    # int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model*, int64_t* token_ids, size_t ntoken)
    lib.llaisysQwen2ModelInfer.restype = c_int64
    lib.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]

    lib.llaisysQwen2ModelInferSample.restype = c_int64
    lib.llaisysQwen2ModelInferSample.argtypes = [c_void_p, POINTER(c_int64), c_size_t, c_float, c_int, c_float, c_uint64]
