from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t
from ctypes import Structure, POINTER, c_size_t, c_float, c_int64, c_int, c_void_p


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


llaisysQwen2Model_t = c_void_p
llaisysQwen2Session_t = c_void_p


def load_qwen2(lib):
    # llaisysQwen2ModelCreate
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t

    # llaisysQwen2ModelDestroy
    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    # llaisysQwen2ModelWeights
    lib.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    # ── 向后兼容：默认 session ────────────────────────────────────────────────
    lib.llaisysQwen2ModelInfer.argtypes = [llaisysQwen2Model_t, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelInferSample.argtypes = [
        llaisysQwen2Model_t, POINTER(c_int64), c_size_t, c_float, c_int, c_float]
    lib.llaisysQwen2ModelInferSample.restype = c_int64

    lib.llaisysQwen2ModelSetCachePos.argtypes = [llaisysQwen2Model_t, c_size_t]
    lib.llaisysQwen2ModelSetCachePos.restype = None

    lib.llaisysQwen2ModelGetCachePos.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelGetCachePos.restype = c_size_t

    lib.llaisysQwen2ModelResetCache.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelResetCache.restype = None

    # ── 多用户 Session API ────────────────────────────────────────────────────
    lib.llaisysQwen2SessionCreate.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2SessionCreate.restype = llaisysQwen2Session_t

    lib.llaisysQwen2SessionDestroy.argtypes = [llaisysQwen2Session_t]
    lib.llaisysQwen2SessionDestroy.restype = None

    lib.llaisysQwen2SessionInfer.argtypes = [
        llaisysQwen2Model_t, llaisysQwen2Session_t, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2SessionInfer.restype = c_int64

    lib.llaisysQwen2SessionInferSample.argtypes = [
        llaisysQwen2Model_t, llaisysQwen2Session_t,
        POINTER(c_int64), c_size_t, c_float, c_int, c_float]
    lib.llaisysQwen2SessionInferSample.restype = c_int64

    lib.llaisysQwen2SessionSetCachePos.argtypes = [llaisysQwen2Session_t, c_size_t]
    lib.llaisysQwen2SessionSetCachePos.restype = None

    lib.llaisysQwen2SessionGetCachePos.argtypes = [llaisysQwen2Session_t]
    lib.llaisysQwen2SessionGetCachePos.restype = c_size_t

    lib.llaisysQwen2SessionResetCache.argtypes = [llaisysQwen2Session_t]
    lib.llaisysQwen2SessionResetCache.restype = None
