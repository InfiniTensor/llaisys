from .libllaisys import LIB_LLAISYS
from .tensor import Tensor
from ctypes import c_float, c_int


class Ops:
    @staticmethod
    def add(c: Tensor, a: Tensor, b: Tensor):
        LIB_LLAISYS.llaisysAdd(c.lib_tensor(), a.lib_tensor(), b.lib_tensor())

    @staticmethod
    def argmax(max_idx: Tensor, max_val: Tensor, vals: Tensor):
        LIB_LLAISYS.llaisysArgmax(max_idx.lib_tensor(), max_val.lib_tensor(), vals.lib_tensor())

    @staticmethod
    def embedding(out: Tensor, index: Tensor, weight: Tensor):
        LIB_LLAISYS.llaisysEmbedding(
            out.lib_tensor(), index.lib_tensor(), weight.lib_tensor()
        )

    @staticmethod
    def linear(out: Tensor, inp: Tensor, weight: Tensor, bias: Tensor):
        LIB_LLAISYS.llaisysLinear(
            out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), bias.lib_tensor()
        )

    @staticmethod
    def rearrange(out: Tensor, inp: Tensor):
        LIB_LLAISYS.llaisysRearrange(out.lib_tensor(), inp.lib_tensor())

    @staticmethod
    def rms_norm(out: Tensor, inp: Tensor, weight: Tensor, eps: float):
        LIB_LLAISYS.llaisysRmsNorm(
            out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), c_float(eps)
        )

    @staticmethod
    def rope(out: Tensor, inp: Tensor, pos_ids: Tensor, theta: float):
        LIB_LLAISYS.llaisysROPE(
            out.lib_tensor(), inp.lib_tensor(), pos_ids.lib_tensor(), c_float(theta)
        )

    @staticmethod
    def self_attention(attn_val: Tensor, q: Tensor, k: Tensor, v: Tensor, scale: float):
        LIB_LLAISYS.llaisysSelfAttention(
            attn_val.lib_tensor(),
            q.lib_tensor(),
            k.lib_tensor(),
            v.lib_tensor(),
            c_float(scale),
        )

    @staticmethod
    def swiglu(out: Tensor, gate: Tensor, up: Tensor):
        LIB_LLAISYS.llaisysSwiGLU(out.lib_tensor(), gate.lib_tensor(), up.lib_tensor())
