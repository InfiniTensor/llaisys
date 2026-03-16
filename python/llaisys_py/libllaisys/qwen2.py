"""ctypes bindings for Qwen2 C API."""
from ctypes import (
    POINTER,
    Structure,
    c_float,
    c_int,
    c_int64,
    c_size_t,
    c_ulonglong,
    c_void_p,
)
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
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
        ("max_batch_size", c_size_t),  # 连续批处理槽位数，1=单序列
        ("tp_rank", c_int),            # 张量并行 rank，默认 0
        ("tp_world_size", c_int),      # 张量并行 world size，1=非分布式
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


LlaisysQwen2Model_t = c_void_p


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model_t

    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelCacheOutputLayerOnCPU.argtypes = [LlaisysQwen2Model_t]
    lib.llaisysQwen2ModelCacheOutputLayerOnCPU.restype = None

    lib.llaisysQwen2ModelCacheAllWeightsOnCPU.argtypes = [LlaisysQwen2Model_t]
    lib.llaisysQwen2ModelCacheAllWeightsOnCPU.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelGetCacheLen.argtypes = [LlaisysQwen2Model_t]
    lib.llaisysQwen2ModelGetCacheLen.restype = c_size_t

    lib.llaisysQwen2ModelGetKVCacheBytes.argtypes = [LlaisysQwen2Model_t, c_size_t]
    lib.llaisysQwen2ModelGetKVCacheBytes.restype = c_size_t

    lib.llaisysQwen2ModelExportKVCache.argtypes = [LlaisysQwen2Model_t, c_void_p]
    lib.llaisysQwen2ModelExportKVCache.restype = None

    lib.llaisysQwen2ModelImportKVCache.argtypes = [
        LlaisysQwen2Model_t,
        c_void_p,
        c_size_t,
    ]
    lib.llaisysQwen2ModelImportKVCache.restype = None

    lib.llaisysQwen2ModelResetKVCache.argtypes = [LlaisysQwen2Model_t]
    lib.llaisysQwen2ModelResetKVCache.restype = None

    lib.llaisysQwen2ModelResetKVCacheSlot.argtypes = [LlaisysQwen2Model_t, c_size_t]
    lib.llaisysQwen2ModelResetKVCacheSlot.restype = None

    lib.llaisysQwen2ModelExportKVCacheSlot.argtypes = [LlaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2ModelExportKVCacheSlot.restype = None

    lib.llaisysQwen2ModelImportKVCacheSlot.argtypes = [
        LlaisysQwen2Model_t,
        c_size_t,
        c_void_p,
        c_size_t,
    ]
    lib.llaisysQwen2ModelImportKVCacheSlot.restype = None

    lib.llaisysQwen2ModelGetCacheLenSlot.argtypes = [LlaisysQwen2Model_t, c_size_t]
    lib.llaisysQwen2ModelGetCacheLenSlot.restype = c_size_t

    lib.llaisysQwen2ModelInferWithSlot.argtypes = [
        LlaisysQwen2Model_t,
        c_size_t,  # slot_id
        POINTER(c_int64),
        c_size_t,
        c_float,
        c_int,
        c_float,
        c_ulonglong,
    ]
    lib.llaisysQwen2ModelInferWithSlot.restype = c_int64

    lib.llaisysQwen2ModelInfer.argtypes = [
        LlaisysQwen2Model_t,
        POINTER(c_int64),
        c_size_t,
        c_float,      # temperature
        c_int,        # top_k
        c_float,      # top_p
        c_ulonglong,  # seed
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelInferHybrid.argtypes = [
        LlaisysQwen2Model_t,
        POINTER(c_int64),
        c_size_t,
        c_float,
        c_int,
        c_float,
        c_ulonglong,
        c_int,        # gpu_up_to_layer: -1=all CPU, 0=only embed GPU, 1=embed+layer0 GPU, ...
    ]
    lib.llaisysQwen2ModelInferHybrid.restype = c_int64

    lib.llaisysQwen2ModelBatchedDecode.argtypes = [
        LlaisysQwen2Model_t,
        POINTER(c_size_t),   # slot_ids
        POINTER(c_int64),    # token_ids
        c_size_t,            # n_batch
        POINTER(c_int64),    # out_next_tokens
        c_float,
        c_int,
        c_float,
        c_ulonglong,
    ]
    lib.llaisysQwen2ModelBatchedDecode.restype = None
