#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
static void rearrange_recursive(
    T *out,
    const T *in,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &out_strides,
    const std::vector<ptrdiff_t> &in_strides,
    size_t dim) {

    if (dim == shape.size() - 1) {
        // Last dimension - copy elements
        for (size_t i = 0; i < shape[dim]; i++) {
            out[i * out_strides[dim]] = in[i * in_strides[dim]];
        }
    } else {
        // Recurse into next dimension
        for (size_t i = 0; i < shape[dim]; i++) {
            rearrange_recursive(
                out + i * out_strides[dim],
                in + i * in_strides[dim],
                shape,
                out_strides,
                in_strides,
                dim + 1);
        }
    }
}

template <typename T>
static void rearrange_(
    T *out,
    const T *in,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &out_strides,
    const std::vector<ptrdiff_t> &in_strides) {

    if (shape.empty()) {
        // Scalar copy
        *out = *in;
        return;
    }

    rearrange_recursive(out, in, shape, out_strides, in_strides, 0);
}

namespace llaisys::ops::cpu {

void rearrange(
    std::byte *out,
    const std::byte *in,
    llaisysDataType_t dtype,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &out_strides,
    const std::vector<ptrdiff_t> &in_strides) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rearrange_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            shape, out_strides, in_strides);
    case LLAISYS_DTYPE_BF16:
        return rearrange_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            shape, out_strides, in_strides);
    case LLAISYS_DTYPE_F16:
        return rearrange_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            shape, out_strides, in_strides);
    case LLAISYS_DTYPE_I64:
        return rearrange_(
            reinterpret_cast<int64_t *>(out),
            reinterpret_cast<const int64_t *>(in),
            shape, out_strides, in_strides);
    case LLAISYS_DTYPE_I32:
        return rearrange_(
            reinterpret_cast<int32_t *>(out),
            reinterpret_cast<const int32_t *>(in),
            shape, out_strides, in_strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
