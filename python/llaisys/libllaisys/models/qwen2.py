import ctypes
from ctypes import POINTER, c_size_t, c_int, c_int64, c_float, c_void_p
from ..llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from ..tensor import llaisysTensor_t

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

LlaisysQwen2Model = c_void_p

def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model

    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelInfer.argtypes = [LlaisysQwen2Model, POINTER(c_int64), c_size_t, c_float, c_int, c_float]
    lib.llaisysQwen2ModelInfer.restype = c_int64
