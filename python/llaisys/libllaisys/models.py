from ctypes import c_void_p, c_int64, c_size_t, c_float, c_int, POINTER, Structure
from . import LIB_LLAISYS

class Qwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),
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

class Qwen2Weights(Structure):
    _fields_ = [
        ("in_embed", c_void_p),
        ("out_embed", c_void_p),
        ("out_norm_w", c_void_p),
        ("attn_norm_w", POINTER(c_void_p)),
        ("attn_q_w", POINTER(c_void_p)),
        ("attn_q_b", POINTER(c_void_p)),
        ("attn_k_w", POINTER(c_void_p)),
        ("attn_k_b", POINTER(c_void_p)),
        ("attn_v_w", POINTER(c_void_p)),
        ("attn_v_b", POINTER(c_void_p)),
        ("attn_o_w", POINTER(c_void_p)),
        ("mlp_norm_w", POINTER(c_void_p)),
        ("mlp_gate_w", POINTER(c_void_p)),
        ("mlp_up_w", POINTER(c_void_p)),
        ("mlp_down_w", POINTER(c_void_p)),
    ]

LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [POINTER(Qwen2Meta), c_int, POINTER(c_int), c_int]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = c_void_p

LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [c_void_p]
LIB_LLAISYS.llaisysQwen2ModelWeights.restype = POINTER(Qwen2Weights)

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = c_int64
