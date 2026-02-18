from ctypes import POINTER, c_void_p, c_size_t, c_int, c_char_p, Structure, c_float, c_int64
from ..llaisys_types import *
from ..tensor import llaisysTensor_t

class Qwen2Meta(Structure):
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

LlaisysQwen2Meta_t = POINTER(Qwen2Meta)
LlaisysQwen2Model_t = c_void_p
LlaisysQwen2Weights_t = POINTER(LlaisysQwen2Weights)


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [LlaisysQwen2Meta_t, llaisysDeviceType_t, POINTER(c_int), c_int]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model_t

    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2modelLoadWeight.argtypes = [LlaisysQwen2Model_t, c_void_p, c_char_p]
    lib.llaisysQwen2modelLoadWeight.restype = None

    lib.llaisysQwen2ModelInfer.argtypes = [LlaisysQwen2Model_t, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = LlaisysQwen2Weights_t
