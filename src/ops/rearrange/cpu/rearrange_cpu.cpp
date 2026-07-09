#include "rearrange_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>

namespace {

template <typename T>
void rearrange_(T *out_base, const T *in_base, const std::vector<size_t> &shape, const std::vector<size_t> &stride_in, const std::vector<size_t> &stride_out,size_t dim, size_t offset_in, size_t offset_out) {
    
    size_t len = shape[dim];
    size_t s_in = stride_in[dim];
    size_t s_out = stride_out[dim];

    if (dim == shape.size() - 1) {
        if (s_in == 1 && s_out == 1) {
            std::memcpy(out_base + offset_out, in_base + offset_in, len * sizeof(T));
        } else {
            for (size_t i = 0; i < len; ++i) {
                out_base[offset_out + i * s_out] = in_base[offset_in + i * s_in];
            }
        }
    } else {
        for (size_t i = 0; i < len; ++i) {
            rearrange_(out_base, in_base, shape, stride_in, stride_out, dim + 1, offset_in + i * s_in, offset_out + i * s_out);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t dtype, const std::vector<size_t> &shape, const std::vector<size_t> &stride_in, const std::vector<size_t> &stride_out) {
    
    if (shape.empty()) {
        size_t size = 0;
        switch (dtype) {
            case LLAISYS_DTYPE_F32: size = 4; break;
            case LLAISYS_DTYPE_BF16: size = 2; break;
            case LLAISYS_DTYPE_F16: size = 2; break;
            case LLAISYS_DTYPE_I64: size = 8; break;
            default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
        std::memcpy(out, in, size);
        return;
    }

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rearrange_(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), shape, stride_in, stride_out, 0, 0, 0);
        break;
    case LLAISYS_DTYPE_BF16:
        rearrange_(reinterpret_cast<llaisys::bf16_t*>(out), reinterpret_cast<const llaisys::bf16_t*>(in), shape, stride_in, stride_out, 0, 0, 0);
        break;
    case LLAISYS_DTYPE_F16:
        rearrange_(reinterpret_cast<llaisys::fp16_t*>(out), reinterpret_cast<const llaisys::fp16_t*>(in), shape, stride_in, stride_out, 0, 0, 0);
        break;
    case LLAISYS_DTYPE_I64:
        rearrange_(reinterpret_cast<int64_t*>(out), reinterpret_cast<const int64_t*>(in), shape, stride_in, stride_out, 0, 0, 0);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu