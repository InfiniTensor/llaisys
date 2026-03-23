from ctypes import Structure, POINTER, CFUNCTYPE, c_size_t, c_int, c_float, c_int64, c_uint32, c_void_p, c_int32

from .llaisys_types import llaisysDeviceType_t, llaisysDataType_t, llaisysStream_t
from .tensor import llaisysTensor_t


class LlaisysQwen2Meta(Structure):
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

class LlaisysSamplingParams(Structure):
    _fields_ = [
        ("top_k", c_int),
        ("top_p", c_float),
        ("temperature", c_float),
        ("seed", c_uint32),
    ]


class LlaisysQwen2KVBlockMeta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("max_tokens", c_size_t),
    ]


LlaisysQwen2Model = c_void_p
LlaisysQwen2KVBlock = c_void_p
LlaisysQwen2KVContext = c_void_p


def load_models(lib):
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

    lib.llaisysQwen2ModelInfer.argtypes = [LlaisysQwen2Model, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelPrefill.argtypes = [LlaisysQwen2Model, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelPrefill.restype = c_int64

    lib.llaisysQwen2ModelStep.argtypes = [LlaisysQwen2Model, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelStep.restype = c_int64
    if hasattr(lib, "llaisysQwen2ModelPrefillPacked"):
        lib.llaisysQwen2ModelPrefillPacked.argtypes = [
            LlaisysQwen2Model,
            POINTER(c_int64),
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
        ]
        lib.llaisysQwen2ModelPrefillPacked.restype = c_int32
    if hasattr(lib, "llaisysQwen2ModelStepPacked"):
        lib.llaisysQwen2ModelStepPacked.argtypes = [
            LlaisysQwen2Model,
            POINTER(c_int64),
            POINTER(c_int64),
            c_size_t,
            POINTER(c_int64),
        ]
        lib.llaisysQwen2ModelStepPacked.restype = c_int32

    if hasattr(lib, "llaisysQwen2ModelPrefillPackedSampling"):
        lib.llaisysQwen2ModelPrefillPackedSampling.argtypes = [
            LlaisysQwen2Model,
            POINTER(c_int64),
            POINTER(c_int64),
            c_size_t,
            POINTER(LlaisysSamplingParams),
            POINTER(c_int64),
        ]
        lib.llaisysQwen2ModelPrefillPackedSampling.restype = c_int32

    if hasattr(lib, "llaisysQwen2ModelStepPackedSampling"):
        lib.llaisysQwen2ModelStepPackedSampling.argtypes = [
            LlaisysQwen2Model,
            POINTER(c_int64),
            POINTER(c_int64),
            c_size_t,
            POINTER(LlaisysSamplingParams),
            POINTER(c_int64),
        ]
        lib.llaisysQwen2ModelStepPackedSampling.restype = c_int32

    lib.llaisysQwen2ModelPrefillSampling.argtypes = [
        LlaisysQwen2Model,
        POINTER(c_int64),
        c_size_t,
        POINTER(LlaisysSamplingParams),
    ]
    lib.llaisysQwen2ModelPrefillSampling.restype = c_int64

    lib.llaisysQwen2ModelStepSampling.argtypes = [
        LlaisysQwen2Model,
        POINTER(c_int64),
        c_size_t,
        POINTER(LlaisysSamplingParams),
    ]
    lib.llaisysQwen2ModelStepSampling.restype = c_int64

    lib.llaisysQwen2ModelInferSampling.argtypes = [
        LlaisysQwen2Model,
        POINTER(c_int64),
        c_size_t,
        POINTER(LlaisysSamplingParams),
    ]
    lib.llaisysQwen2ModelInferSampling.restype = c_int64

    lib.llaisysQwen2ModelInferSamplingEx.argtypes = [
        LlaisysQwen2Model,
        POINTER(c_int64),
        c_size_t,
        c_int,
        c_float,
        c_float,
        c_uint32,
    ]
    lib.llaisysQwen2ModelInferSamplingEx.restype = c_int64

    lib.llaisysQwen2ModelResetKVCache.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelResetKVCache.restype = None

    lib.llaisysQwen2ModelSetKVCacheEnabled.argtypes = [LlaisysQwen2Model, c_int]
    lib.llaisysQwen2ModelSetKVCacheEnabled.restype = None

    # Experimental KV block/context APIs
    lib.llaisysQwen2KVBlockCreate.argtypes = [
        POINTER(LlaisysQwen2KVBlockMeta),
        llaisysDeviceType_t,
        c_int,
    ]
    lib.llaisysQwen2KVBlockCreate.restype = LlaisysQwen2KVBlock
    lib.llaisysQwen2KVBlockRetain.argtypes = [LlaisysQwen2KVBlock]
    lib.llaisysQwen2KVBlockRetain.restype = None
    lib.llaisysQwen2KVBlockRelease.argtypes = [LlaisysQwen2KVBlock]
    lib.llaisysQwen2KVBlockRelease.restype = None
    lib.llaisysQwen2KVBlockSetTokenCount.argtypes = [LlaisysQwen2KVBlock, c_size_t]
    lib.llaisysQwen2KVBlockSetTokenCount.restype = c_int32
    lib.llaisysQwen2KVBlockTokenCount.argtypes = [LlaisysQwen2KVBlock]
    lib.llaisysQwen2KVBlockTokenCount.restype = c_size_t
    lib.llaisysQwen2KVBlockKeyTensor.argtypes = [LlaisysQwen2KVBlock, c_size_t]
    lib.llaisysQwen2KVBlockKeyTensor.restype = llaisysTensor_t
    lib.llaisysQwen2KVBlockValueTensor.argtypes = [LlaisysQwen2KVBlock, c_size_t]
    lib.llaisysQwen2KVBlockValueTensor.restype = llaisysTensor_t

    lib.llaisysQwen2KVContextCreate.argtypes = [
        llaisysDataType_t,
        llaisysDeviceType_t,
        c_int,
        c_size_t,
        c_size_t,
        c_size_t,
        c_size_t,
    ]
    lib.llaisysQwen2KVContextCreate.restype = LlaisysQwen2KVContext
    lib.llaisysQwen2KVContextRetain.argtypes = [LlaisysQwen2KVContext]
    lib.llaisysQwen2KVContextRetain.restype = None
    lib.llaisysQwen2KVContextRelease.argtypes = [LlaisysQwen2KVContext]
    lib.llaisysQwen2KVContextRelease.restype = None
    lib.llaisysQwen2KVContextAttachBlock.argtypes = [LlaisysQwen2KVContext, LlaisysQwen2KVBlock]
    lib.llaisysQwen2KVContextAttachBlock.restype = c_int32
    lib.llaisysQwen2KVContextDetachAll.argtypes = [LlaisysQwen2KVContext]
    lib.llaisysQwen2KVContextDetachAll.restype = None
    lib.llaisysQwen2KVContextBlockCount.argtypes = [LlaisysQwen2KVContext]
    lib.llaisysQwen2KVContextBlockCount.restype = c_size_t
    lib.llaisysQwen2KVContextTokenCount.argtypes = [LlaisysQwen2KVContext]
    lib.llaisysQwen2KVContextTokenCount.restype = c_size_t

    lib.llaisysQwen2ModelSetKVContext.argtypes = [LlaisysQwen2Model, LlaisysQwen2KVContext]
    lib.llaisysQwen2ModelSetKVContext.restype = c_int32
    lib.llaisysQwen2ModelGetKVContext.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelGetKVContext.restype = LlaisysQwen2KVContext
    lib.llaisysQwen2ModelExportKVContext.argtypes = [LlaisysQwen2Model, LlaisysQwen2KVContext, c_size_t]
    lib.llaisysQwen2ModelExportKVContext.restype = c_int32

    if hasattr(lib, "llaisysQwen2ModelSetTensorParallel"):
        lib.llaisysQwen2ModelSetTensorParallel.argtypes = [LlaisysQwen2Model, c_void_p, c_void_p, c_int]
        lib.llaisysQwen2ModelSetTensorParallel.restype = c_int32


# --- Comm API ctypes ---

llaisysComm_t = c_void_p

LLAISYS_COMM_UNIQUE_ID_MAX_SIZE = 128

comm_init_api = CFUNCTYPE(c_int, POINTER(llaisysComm_t), c_int, c_int, c_void_p)
comm_destroy_api = CFUNCTYPE(None, llaisysComm_t)
comm_get_rank_api = CFUNCTYPE(c_int, llaisysComm_t)
comm_get_size_api = CFUNCTYPE(c_int, llaisysComm_t)
comm_allreduce_api = CFUNCTYPE(
    None, c_void_p, c_void_p, c_size_t, c_int, c_int, llaisysComm_t, llaisysStream_t,
)
comm_broadcast_api = CFUNCTYPE(
    None, c_void_p, c_size_t, c_int, c_int, llaisysComm_t, llaisysStream_t,
)
comm_send_api = CFUNCTYPE(
    None, c_void_p, c_size_t, c_int, c_int, llaisysComm_t, llaisysStream_t,
)
comm_recv_api = CFUNCTYPE(
    None, c_void_p, c_size_t, c_int, c_int, llaisysComm_t, llaisysStream_t,
)


class LlaisysCommAPI(Structure):
    _fields_ = [
        ("init", comm_init_api),
        ("destroy", comm_destroy_api),
        ("get_rank", comm_get_rank_api),
        ("get_size", comm_get_size_api),
        ("allreduce", comm_allreduce_api),
        ("broadcast", comm_broadcast_api),
        ("send", comm_send_api),
        ("recv", comm_recv_api),
    ]


def load_comm(lib):
    if hasattr(lib, "llaisysGetCommAPI"):
        lib.llaisysGetCommAPI.argtypes = [c_int]
        lib.llaisysGetCommAPI.restype = POINTER(LlaisysCommAPI)
    if hasattr(lib, "llaisysCommGenerateUniqueId"):
        lib.llaisysCommGenerateUniqueId.argtypes = [c_int, c_void_p, POINTER(c_size_t)]
        lib.llaisysCommGenerateUniqueId.restype = c_int


__all__ = [
    "LlaisysQwen2Meta",
    "LlaisysQwen2Weights",
    "LlaisysSamplingParams",
    "LlaisysQwen2KVBlockMeta",
    "LlaisysQwen2Model",
    "LlaisysQwen2KVBlock",
    "LlaisysQwen2KVContext",
    "load_models",
    "LlaisysCommAPI",
    "llaisysComm_t",
    "LLAISYS_COMM_UNIQUE_ID_MAX_SIZE",
    "load_comm",
]
