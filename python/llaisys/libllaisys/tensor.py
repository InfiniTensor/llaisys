from ctypes import POINTER, c_uint8, c_void_p, c_size_t, c_ssize_t, c_int
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t

# Handle type
llaisysTensor_t = c_void_p


def load_tensor(lib):
    lib.tensorCreate.argtypes = [
        POINTER(c_size_t),  # shape
        c_size_t,  # ndim
        llaisysDataType_t,  # dtype
        llaisysDeviceType_t,  # device_type
        c_int,  # device_id
    ]
    lib.tensorCreate.restype = llaisysTensor_t

    # Function: tensorDestroy
    lib.tensorDestroy.argtypes = [llaisysTensor_t]
    lib.tensorDestroy.restype = None

    # Function: tensorGetData
    lib.tensorGetData.argtypes = [llaisysTensor_t]
    lib.tensorGetData.restype = c_void_p

    # Function: tensorGetNdim
    lib.tensorGetNdim.argtypes = [llaisysTensor_t]
    lib.tensorGetNdim.restype = c_size_t

    # Function: tensorGetShape
    lib.tensorGetShape.argtypes = [llaisysTensor_t, POINTER(c_size_t)]
    lib.tensorGetShape.restype = None

    # Function: tensorGetStrides
    lib.tensorGetStrides.argtypes = [llaisysTensor_t, POINTER(c_ssize_t)]
    lib.tensorGetStrides.restype = None

    # Function: tensorGetDataType
    lib.tensorGetDataType.argtypes = [llaisysTensor_t]
    lib.tensorGetDataType.restype = llaisysDataType_t

    # Function: tensorGetDeviceType
    lib.tensorGetDeviceType.argtypes = [llaisysTensor_t]
    lib.tensorGetDeviceType.restype = llaisysDeviceType_t

    # Function: tensorGetDeviceId
    lib.tensorGetDeviceId.argtypes = [llaisysTensor_t]
    lib.tensorGetDeviceId.restype = c_int

    # Function: tensorDebug
    lib.tensorDebug.argtypes = [llaisysTensor_t]
    lib.tensorDebug.restype = None

    # Function: tensorIsContiguous
    lib.tensorIsContiguous.argtypes = [llaisysTensor_t]
    lib.tensorIsContiguous.restype = c_uint8

    # Function: tensorLoad
    lib.tensorLoad.argtypes = [llaisysTensor_t, c_void_p]
    lib.tensorLoad.restype = None

    # Function: tensorView(llaisysTensor_t tensor, size_t *shape);
    lib.tensorView.argtypes = [llaisysTensor_t, POINTER(c_size_t), c_size_t]
    lib.tensorView.restype = llaisysTensor_t

    # Function: tensorPermute(llaisysTensor_t tensor, size_t *order);
    lib.tensorPermute.argtypes = [llaisysTensor_t, POINTER(c_size_t)]
    lib.tensorPermute.restype = llaisysTensor_t

    # Function: tensorSlice(llaisysTensor_t tensor,
    #                     size_t dim, size_t start, size_t end);
    lib.tensorSlice.argtypes = [
        llaisysTensor_t,  # tensor handle
        c_size_t,  # dim  : which axis to slice
        c_size_t,  # start: inclusive
        c_size_t,  # end  : exclusive
    ]
    lib.tensorSlice.restype = llaisysTensor_t
