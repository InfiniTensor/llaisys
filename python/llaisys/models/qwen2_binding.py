import ctypes
from ctypes import c_size_t, c_int, c_float, c_void_p, c_int64, POINTER, Structure, c_char_p
from ..libllaisys.llaisys_types import DataType, llaisysDataType_t, llaisysDeviceType_t

class Qwen2MetaCStruct(Structure):
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

# Opaque pointer handle
LlaisysQwen2ModelHandle = c_void_p

def register_qwen2_lib(lib):
    if hasattr(lib, "llaisysQwen2ModelCreate"):
        # Create
        lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2ModelHandle
        lib.llaisysQwen2ModelCreate.argtypes = [
            POINTER(Qwen2MetaCStruct),
            llaisysDeviceType_t,
            POINTER(c_int),     # device_ids
            c_int               # ndev
        ]

        # Destroy
        lib.llaisysQwen2ModelDestroy.restype = None
        lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2ModelHandle]

        # Load Weight
        lib.llaisysQwen2LoadWeight.restype = None
        lib.llaisysQwen2LoadWeight.argtypes = [
            LlaisysQwen2ModelHandle,
            c_char_p,          # name
            c_void_p,          # data
            POINTER(c_size_t), # shape
            c_size_t,          # ndim
            llaisysDataType_t  # dtype
        ]

        # Infer
        lib.llaisysQwen2ModelInfer.restype = c_int64
        lib.llaisysQwen2ModelInfer.argtypes = [
            LlaisysQwen2ModelHandle,
            POINTER(c_int64), # input_ids_ptr
            c_size_t,         # seq_len
            c_size_t          # start_pos
        ]
