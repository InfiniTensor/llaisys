import ctypes
from ctypes import c_void_p, c_size_t, c_int, Structure, CFUNCTYPE
from .llaisys_types import *
from .tensor import llaisysTensor_t

# Define function pointer types
get_device_count_api = CFUNCTYPE(c_int)
set_device_api = CFUNCTYPE(None, c_int)
device_synchronize_api = CFUNCTYPE(None)

create_stream_api = CFUNCTYPE(llaisysStream_t)
destroy_stream_api = CFUNCTYPE(None, llaisysStream_t)
stream_synchronize_api = CFUNCTYPE(None, llaisysStream_t)

malloc_device_api = CFUNCTYPE(c_void_p, c_size_t)
free_device_api = CFUNCTYPE(None, c_void_p)
malloc_host_api = CFUNCTYPE(c_void_p, c_size_t)
free_host_api = CFUNCTYPE(None, c_void_p)

memcpy_sync_api = CFUNCTYPE(None, c_void_p, c_void_p, c_size_t, llaisysMemcpyKind_t)
memcpy_async_api = CFUNCTYPE(None, c_void_p, c_void_p, c_size_t, llaisysMemcpyKind_t, llaisysStream_t)


# Define the struct matching LlaisysRuntimeAPI
class LlaisysRuntimeAPI(Structure):
    _fields_ = [
        ("get_device_count", get_device_count_api),
        ("set_device", set_device_api),
        ("device_synchronize", device_synchronize_api),
        ("create_stream", create_stream_api),
        ("destroy_stream", destroy_stream_api),
        ("stream_synchronize", stream_synchronize_api),
        ("malloc_device", malloc_device_api),
        ("free_device", free_device_api),
        ("malloc_host", malloc_host_api),
        ("free_host", free_host_api),
        ("memcpy_sync", memcpy_sync_api),
        ("memcpy_async", memcpy_async_api),
    ]


# Load shared library
def load_runtime(lib):
    # Declare API function prototypes
    lib.llaisysGetRuntimeAPI.argtypes = [llaisysDeviceType_t]
    lib.llaisysGetRuntimeAPI.restype = ctypes.POINTER(LlaisysRuntimeAPI)

    lib.llaisysSetContextRuntime.argtypes = [llaisysDeviceType_t, c_int]
    lib.llaisysSetContextRuntime.restype = None

    lib.llaisysInitDistributed.argtypes = [c_int, c_int]
    lib.llaisysInitDistributed.restype = None

    lib.llaisysFinalizeDistributed.argtypes = []
    lib.llaisysFinalizeDistributed.restype = None

    lib.llaisysDistributedIsInitialized.argtypes = []
    lib.llaisysDistributedIsInitialized.restype = ctypes.c_uint8

    lib.llaisysDistributedRank.argtypes = []
    lib.llaisysDistributedRank.restype = c_int

    lib.llaisysDistributedWorldSize.argtypes = []
    lib.llaisysDistributedWorldSize.restype = c_int

    lib.llaisysDistAllReduce.argtypes = [llaisysTensor_t]
    lib.llaisysDistAllReduce.restype = None

    lib.llaisysDistAllGather.argtypes = [llaisysTensor_t]
    lib.llaisysDistAllGather.restype = llaisysTensor_t

    lib.llaisysDistBroadcast.argtypes = [llaisysTensor_t, c_int]
    lib.llaisysDistBroadcast.restype = None

    lib.llaisysDistBarrier.argtypes = []
    lib.llaisysDistBarrier.restype = None
