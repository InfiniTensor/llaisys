import ctypes
from enum import IntEnum

import torch


# Device Type enum
class DeviceType(IntEnum):
    CPU = 0
    NVIDIA = 1
    COUNT = 2


llaisysDeviceType_t = ctypes.c_int


# Data Type enum
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


def to_data_type(dtype: str) -> llaisysDataType_t:
    if dtype == "bool":
        return DataType.BOOL
    elif dtype == "int8":
        return DataType.I8
    elif dtype == "int16":
        return DataType.I16
    elif dtype == "int32":
        return DataType.I32
    elif dtype == "int64":
        return DataType.I64
    elif dtype == "uint8":
        return DataType.U8
    elif dtype == "uint16":
        return DataType.U16
    elif dtype == "uint32":
        return DataType.U32
    elif dtype == "uint64":
        return DataType.U64
    elif dtype == "bfloat16":
        return DataType.BF16
    elif dtype == "float16":
        return DataType.F16
    elif dtype == "float32":
        return DataType.F32
    elif dtype == "float64":
        return DataType.F64
    else:
        print("to_data_type():  dtype is", dtype)
        raise ValueError("Unsupported data type")


def to_torch_type(dtype: DataType) -> torch.dtype:
    if dtype == DataType.BOOL:
        return torch.bool
    elif dtype == DataType.I8:
        return torch.int8
    elif dtype == DataType.I16:
        return torch.int16
    elif dtype == DataType.I32:
        return torch.int32
    elif dtype == DataType.I64:
        return torch.int64
    elif dtype == DataType.U8:
        return torch.uint8
    elif dtype == DataType.U16:
        return torch.uint16
    elif dtype == DataType.U32:
        return torch.uint32
    elif dtype == DataType.U64:
        return torch.uint64
    elif dtype == DataType.BF16:
        return torch.bfloat16
    elif dtype == DataType.F16:
        return torch.float16
    elif dtype == DataType.F32:
        return torch.float32
    elif dtype == DataType.F64:
        return torch.float64
    else:
        print("to_torch_type():  dtype is", dtype)
        raise ValueError("Unsupported data type")


# Memory Copy Kind enum
class MemcpyKind(IntEnum):
    H2H = 0
    H2D = 1
    D2H = 2
    D2D = 3


llaisysMemcpyKind_t = ctypes.c_int

# Stream type (opaque pointer)
llaisysStream_t = ctypes.c_void_p

__all__ = [
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysDataType_t",
    "DataType",
    "to_data_type",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
]
