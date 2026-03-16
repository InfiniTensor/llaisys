from .tensor import llaisysTensor_t
from ctypes import c_float, c_int, c_ulonglong

def load_ops(lib):
    lib.llaisysAdd.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysAdd.restype = None

    lib.llaisysArgmax.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysArgmax.restype = None

    lib.llaisysEmbedding.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysEmbedding.restype = None

    lib.llaisysLinear.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysLinear.restype = None

    lib.llaisysRearrange.argtypes = [llaisysTensor_t, llaisysTensor_t]
    lib.llaisysRearrange.restype = None

    lib.llaisysRmsNorm.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, c_float]
    lib.llaisysRmsNorm.restype = None

    lib.llaisysROPE.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, c_float]
    lib.llaisysROPE.restype = None

    lib.llaisysSelfAttention.argtypes = [
        llaisysTensor_t,  # attn_val
        llaisysTensor_t,  # q
        llaisysTensor_t,  # k
        llaisysTensor_t,  # v
        c_float    # scale
    ]
    lib.llaisysSelfAttention.restype = None

    lib.llaisysSwiGLU.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysSwiGLU.restype = None

    lib.llaisysSample.argtypes = [
        llaisysTensor_t,  # out_idx
        llaisysTensor_t,  # logits
        c_float,         # temperature
        c_int,           # top_k
        c_float,         # top_p
        c_ulonglong,     # seed
    ]
    lib.llaisysSample.restype = None
