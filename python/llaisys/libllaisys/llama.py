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


class LlaisysLlamaMeta(Structure):
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


class LlaisysLlamaWeights(Structure):
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


def load_llama_api(lib):
    lib.llaisysLlamaModelCreate.restype = c_void_p
    lib.llaisysLlamaModelCreate.argtypes = [POINTER(LlaisysLlamaMeta), llaisysDeviceType_t, POINTER(c_int), c_int]

    lib.llaisysLlamaModelDestroy.restype = None
    lib.llaisysLlamaModelDestroy.argtypes = [c_void_p]

    lib.llaisysLlamaModelWeights.restype = POINTER(LlaisysLlamaWeights)
    lib.llaisysLlamaModelWeights.argtypes = [c_void_p]

    lib.llaisysLlamaModelReset.restype = None
    lib.llaisysLlamaModelReset.argtypes = [c_void_p]

    lib.llaisysLlamaModelInfer.restype = c_int64
    lib.llaisysLlamaModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]

    lib.llaisysLlamaModelInferSample.restype = c_int64
    lib.llaisysLlamaModelInferSample.argtypes = [c_void_p, POINTER(c_int64), c_size_t, c_float, c_int, c_float, c_uint64]
