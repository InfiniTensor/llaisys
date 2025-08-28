# python/llaisys/libllaisys/qwen2.py
from ctypes import (
    c_int, c_size_t, c_longlong, c_void_p, c_char_p, c_float,
    POINTER, Structure
)
from . import LIB_LLAISYS  # 复用统一加载的 DLL 句柄

class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype",     c_int),
        ("nlayer",    c_size_t),
        ("hs",        c_size_t),
        ("nh",        c_size_t),
        ("nkvh",      c_size_t),
        ("dh",        c_size_t),
        ("di",        c_size_t),
        ("maxseq",    c_size_t),
        ("voc",       c_size_t),
        ("epsilon",   c_float),     # <<< 修正
        ("theta",     c_float),     # <<< 修正
        ("end_token", c_longlong),
    ]

# C 函数签名
LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [POINTER(LlaisysQwen2Meta), c_int, POINTER(c_int), c_int]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype  = c_void_p

LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype  = None

LIB_LLAISYS.llaisysQwen2ModelLoadNamedWeight.argtypes = [
    c_void_p, c_char_p, c_void_p, POINTER(c_size_t), c_size_t, c_int
]
LIB_LLAISYS.llaisysQwen2ModelLoadNamedWeight.restype = c_int

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_longlong), c_size_t]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype  = c_longlong
