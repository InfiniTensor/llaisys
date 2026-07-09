#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
static void rearrange_(T *out, const T *in, const size_t *shape, const ptrdiff_t *out_strides, 
    const ptrdiff_t *in_strides, const size_t numel, size_t ndim) {
    for (size_t i = 0; i < numel; ++i) {
        size_t tmp = i;
        ptrdiff_t in_offset = 0;
        ptrdiff_t out_offset = 0;
        for (size_t d = ndim; d-- > 0;) {
            // cal index from d-1 to 0;
            size_t idx = tmp % shape[d];
            tmp /= shape[d];
            // based strides to cal offset
            in_offset += static_cast<ptrdiff_t>(idx) * in_strides[d];
            out_offset += static_cast<ptrdiff_t>(idx) * out_strides[d];
        }
        out[out_offset] = in[in_offset];
    }
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, const size_t *shape,
               const ptrdiff_t *out_strides, const ptrdiff_t *in_strides, size_t numel, size_t ndim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_(reinterpret_cast<float *>(out), 
        reinterpret_cast<const float *>(in),
        shape, out_strides, in_strides, numel, ndim);
    case LLAISYS_DTYPE_BF16:
        return rearrange_(reinterpret_cast<llaisys::bf16_t *>(out), 
        reinterpret_cast<const llaisys::bf16_t *>(in),
        shape, out_strides, in_strides, numel, ndim);
    case LLAISYS_DTYPE_F16:
        return rearrange_(reinterpret_cast<llaisys::fp16_t *>(out), 
        reinterpret_cast<const llaisys::fp16_t *>(in),
        shape, out_strides, in_strides, numel, ndim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
