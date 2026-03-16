from ctypes import (
    Structure,
    POINTER,
    c_int,
    c_size_t,
    c_float,
    c_int64,
    c_void_p,
    c_char_p,
    c_uint64,
)

from .llaisys_types import llaisysDeviceType_t, llaisysDataType_t


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


def load_qwen2(lib):
    # llasiysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
    if hasattr(lib, 'llaisysQwen2ModelCreate'):
        lib.llaisysQwen2ModelCreate.argtypes = [POINTER(LlaisysQwen2Meta), llaisysDeviceType_t, POINTER(c_int), c_int]
        lib.llaisysQwen2ModelCreate.restype = c_void_p
    else:
        print('[libllaisys.qwen2] Warning: llaisysQwen2ModelCreate not found in shared lib')

    # void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);
    if hasattr(lib, 'llaisysQwen2ModelDestroy'):
        lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
        lib.llaisysQwen2ModelDestroy.restype = None

    # struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);
    if hasattr(lib, 'llaisysQwen2ModelWeights'):
        lib.llaisysQwen2ModelWeights.argtypes = [c_void_p]
        lib.llaisysQwen2ModelWeights.restype = c_void_p

    # int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);
    if hasattr(lib, 'llaisysQwen2ModelInfer'):
        lib.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
        lib.llaisysQwen2ModelInfer.restype = c_int64
    else:
        print('[libllaisys.qwen2] Warning: llaisysQwen2ModelInfer not found in shared lib')

    # int64_t llaisysQwen2ModelInferSample(struct LlaisysQwen2Model * model,
    #                                      int64_t * token_ids, size_t ntoken,
    #                                      float temperature, size_t top_k, float top_p, uint64_t seed);
    if hasattr(lib, 'llaisysQwen2ModelInferSample'):
        lib.llaisysQwen2ModelInferSample.argtypes = [
            c_void_p,
            POINTER(c_int64),
            c_size_t,
            c_float,
            c_size_t,
            c_float,
            c_uint64,
        ]
        lib.llaisysQwen2ModelInferSample.restype = c_int64

    # int llaisysQwen2ModelSetWeight(struct LlaisysQwen2Model * model, const char * name, llaisysTensor_t tensor);
    if hasattr(lib, 'llaisysQwen2ModelSetWeight'):
        lib.llaisysQwen2ModelSetWeight.argtypes = [c_void_p, c_char_p, c_void_p]
        lib.llaisysQwen2ModelSetWeight.restype = c_int

    # int llaisysQwen2ModelFinalize(struct LlaisysQwen2Model * model);
    if hasattr(lib, 'llaisysQwen2ModelFinalize'):
        lib.llaisysQwen2ModelFinalize.argtypes = [c_void_p]
        lib.llaisysQwen2ModelFinalize.restype = c_int

    # uint8_t llaisysQwen2ModelHasWeight(struct LlaisysQwen2Model * model, const char * name);
    if hasattr(lib, 'llaisysQwen2ModelHasWeight'):
        lib.llaisysQwen2ModelHasWeight.argtypes = [c_void_p, c_char_p]
        lib.llaisysQwen2ModelHasWeight.restype = c_int

    # KV cache prototypes
    if hasattr(lib, 'llaisysQwen2KVCreat'):
        lib.llaisysQwen2KVCreat.argtypes = [c_void_p, c_size_t]
        lib.llaisysQwen2KVCreat.restype = c_void_p

    if hasattr(lib, 'llaisysQwen2KVDestroy'):
        lib.llaisysQwen2KVDestroy.argtypes = [c_void_p]
        lib.llaisysQwen2KVDestroy.restype = None

    if hasattr(lib, 'llaisysQwen2KVAppend'):
        lib.llaisysQwen2KVAppend.argtypes = [c_void_p, c_void_p, c_void_p]
        lib.llaisysQwen2KVAppend.restype = c_int

    if hasattr(lib, 'llaisysQwen2KVLen'):
        lib.llaisysQwen2KVLen.argtypes = [c_void_p]
        lib.llaisysQwen2KVLen.restype = c_size_t


__all__ = ["LlaisysQwen2Meta", "load_qwen2"]
