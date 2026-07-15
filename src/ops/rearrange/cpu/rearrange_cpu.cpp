#include "rearrange_cpu.hpp"
#include "../../../utils.hpp"

#include <cstddef>
#include <cstring>

namespace {

size_t shape_numel(const std::vector<size_t> &shape) {
    size_t numel = 1;
    for (size_t dim : shape) {
        numel *= dim;
    }
    return numel;
}

bool is_contiguous(const std::vector<size_t> &shape, const std::vector<size_t> &stride) {
    if (shape.size() != stride.size()) {
        return false;
    }

    size_t expected = 1;
    for (std::ptrdiff_t dim = static_cast<std::ptrdiff_t>(shape.size()) - 1; dim >= 0; --dim) {
        if (stride[dim] != expected) {
            return false;
        }
        expected *= shape[dim];
    }
    return true;
}

template <typename T>
void rearrange_inner(T *out_base, const T *in_base, const std::vector<size_t> &shape, const std::vector<size_t> &stride_in, const std::vector<size_t> &stride_out, size_t dim, size_t offset_in, size_t offset_out) {
    const size_t len = shape[dim];
    const size_t s_in = stride_in[dim];
    const size_t s_out = stride_out[dim];

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
            rearrange_inner(out_base, in_base, shape, stride_in, stride_out, dim + 1, offset_in + i * s_in, offset_out + i * s_out);
        }
    }
}

template <typename T>
void rearrange_dispatch(T *out_base, const T *in_base, const std::vector<size_t> &shape, const std::vector<size_t> &stride_in, const std::vector<size_t> &stride_out) {
    if (shape.empty()) {
        return;
    }
    if (is_contiguous(shape, stride_in) && is_contiguous(shape, stride_out)) {
        std::memcpy(out_base, in_base, shape_numel(shape) * sizeof(T));
        return;
    }
    if (shape.size() == 1) {
        rearrange_inner(out_base, in_base, shape, stride_in, stride_out, 0, 0, 0);
        return;
    }

    const size_t len0 = shape[0];
    const size_t s_in0 = stride_in[0];
    const size_t s_out0 = stride_out[0];

#pragma omp parallel for schedule(static) if (len0 >= 4)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(len0); ++i) {
        rearrange_inner(out_base, in_base, shape, stride_in, stride_out, 1, static_cast<size_t>(i) * s_in0, static_cast<size_t>(i) * s_out0);
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
        rearrange_dispatch(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(in), shape, stride_in, stride_out);
        break;
    case LLAISYS_DTYPE_BF16:
        rearrange_dispatch(reinterpret_cast<llaisys::bf16_t*>(out), reinterpret_cast<const llaisys::bf16_t*>(in), shape, stride_in, stride_out);
        break;
    case LLAISYS_DTYPE_F16:
        rearrange_dispatch(reinterpret_cast<llaisys::fp16_t*>(out), reinterpret_cast<const llaisys::fp16_t*>(in), shape, stride_in, stride_out);
        break;
    case LLAISYS_DTYPE_I64:
        rearrange_dispatch(reinterpret_cast<int64_t*>(out), reinterpret_cast<const int64_t*>(in), shape, stride_in, stride_out);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
