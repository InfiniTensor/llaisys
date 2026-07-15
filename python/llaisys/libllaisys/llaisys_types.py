import ctypes
from enum import IntEnum


class DeviceType(IntEnum):
    CPU = 0
    NVIDIA = 1
    COUNT = 2


llaisysDeviceType_t = ctypes.c_int


class DataType(IntEnum):
    INVALID = 0
    BYTE = 1
    BOOL = 2
    I8 = 3
    I16 = 4
    I32 = 5
    I64 = 6
    U8 = 7
    U16 = 8
    U32 = 9
    U64 = 10
    F8 = 11
    F16 = 12
    F32 = 13
    F64 = 14
    C16 = 15
    C32 = 16
    C64 = 17
    C128 = 18
    BF16 = 19


llaisysDataType_t = ctypes.c_int


class MemcpyKind(IntEnum):
    H2H = 0
    H2D = 1
    D2H = 2
    D2D = 3


llaisysMemcpyKind_t = ctypes.c_int
llaisysTensor_t = ctypes.c_void_p

class LlaisysQwen2Model(ctypes.Structure):
    pass
llaisysQwen2ModelHandle = ctypes.POINTER(LlaisysQwen2Model)
llaisysQwen2Weights_p = ctypes.c_void_p 

llaisysStream_t = ctypes.c_void_p

__all__ = [
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysDataType_t",
    "DataType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
]
