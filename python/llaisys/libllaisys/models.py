from .tensor import llaisysTensor_t
from .llaisys_types import (
    llaisysDataType_t,
    llaisysDeviceType_t,
    DataType,
    DeviceType,
)
from ctypes import (
    c_float,
    c_int64,
    c_size_t,
    POINTER,
    Structure,
    c_int,
    c_void_p,
)


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),  # llaisysDataType_t
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


llaisysQwen2Model_t = c_void_p


def load_models(lib):
    # Meta structure
    lib.LlaisysQwen2Meta = LlaisysQwen2Meta
    lib.LlaisysQwen2Weights = LlaisysQwen2Weights

    # llaisysQwen2ModelCreate
    # argtypes用于指定函数参数类型，restype用于指定返回类型
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        c_int,  # llaisysDeviceType_t
        POINTER(c_int),  # int *device_ids
        c_int,  # int ndevice
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t

    # llaisysQwen2ModelDestroy
    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    # llaisysQwen2ModelWeights
    lib.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    # llaisysQwen2ModelResetCache
    lib.llaisysQwen2ModelResetCache.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelResetCache.restype = None

    # llaisysQwen2ModelInfer
    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t,
        POINTER(c_int64),  # int64_t *token_ids
        c_size_t,  # size_t ntoken
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    # llaisysQwen2ModelInferSample
    lib.llaisysQwen2ModelInferSample.argtypes = [
        llaisysQwen2Model_t,
        POINTER(c_int64),  # int64_t *token_ids
        c_size_t,  # size_t ntoken
        c_int,  # int top_k
        c_float,  # float top_p
        c_float,  # float temperature
    ]
    lib.llaisysQwen2ModelInferSample.restype = c_int64
