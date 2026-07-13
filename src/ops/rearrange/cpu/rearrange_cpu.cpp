#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void rearrange_(T *out, const T *in,
                const std::vector<size_t> &shape,
                const std::vector<ptrdiff_t> &out_strides,
                const std::vector<ptrdiff_t> &in_strides) {
    size_t ndim = shape.size();
    
    if (ndim == 0) {
        out[0] = in[0];
        return;
    }
    
    // Calculate total number of elements
    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        numel *= shape[i];
    }
    
    // Use indices to iterate through all elements
    std::vector<size_t> indices(ndim, 0);
    
    for (size_t i = 0; i < numel; ++i) {
        // Calculate source and destination offsets
        ptrdiff_t out_offset = 0;
        ptrdiff_t in_offset = 0;
        for (size_t d = 0; d < ndim; ++d) {
            out_offset += indices[d] * out_strides[d];
            in_offset += indices[d] * in_strides[d];
        }
        
        // Copy element
        out[out_offset] = in[in_offset];
        
        // Increment indices (like counting in mixed radix)
        for (ptrdiff_t d = ndim - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < shape[d]) {
                break;
            }
            indices[d] = 0;
        }
    }
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides,
               llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_(reinterpret_cast<float *>(out),
                          reinterpret_cast<const float *>(in),
                          shape, out_strides, in_strides);
    case LLAISYS_DTYPE_BF16:
        return rearrange_(reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const llaisys::bf16_t *>(in),
                          shape, out_strides, in_strides);
    case LLAISYS_DTYPE_F16:
        return rearrange_(reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const llaisys::fp16_t *>(in),
                          shape, out_strides, in_strides);
    case LLAISYS_DTYPE_I64:
        return rearrange_(reinterpret_cast<int64_t *>(out),
                          reinterpret_cast<const int64_t *>(in),
                          shape, out_strides, in_strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
