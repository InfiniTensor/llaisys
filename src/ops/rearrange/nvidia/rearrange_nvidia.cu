#include "rearrange_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace detail {
inline void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}
}

template <typename T>
__global__ void rearrange_kernel(T *out, const T *in, const size_t *shape, const ptrdiff_t *out_strides, const ptrdiff_t *in_strides,
                                 size_t numel, size_t ndim) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    size_t tmp = idx;
    ptrdiff_t in_offset = 0;
    ptrdiff_t out_offset = 0;
    for (size_t d = ndim; d-- > 0;) {
        size_t id = tmp % shape[d];
        tmp /= shape[d];
        in_offset += static_cast<ptrdiff_t>(id) * in_strides[d];
        out_offset += static_cast<ptrdiff_t>(id) * out_strides[d];
    }
    out[out_offset] = in[in_offset];
}

void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, const size_t *shape, const ptrdiff_t *out_strides,
               const ptrdiff_t *in_strides, size_t numel, size_t ndim) {
    auto &runtime = llaisys::core::context().runtime();
    size_t shape_bytes = ndim * sizeof(size_t);
    size_t stride_bytes = ndim * sizeof(ptrdiff_t);
    auto d_shape = static_cast<size_t *>(runtime.api()->malloc_device(shape_bytes));
    auto d_out_strides = static_cast<ptrdiff_t *>(runtime.api()->malloc_device(stride_bytes));
    auto d_in_strides = static_cast<ptrdiff_t *>(runtime.api()->malloc_device(stride_bytes));

    runtime.api()->memcpy_sync(d_shape, shape, shape_bytes, LLAISYS_MEMCPY_H2D);
    runtime.api()->memcpy_sync(d_out_strides, out_strides, stride_bytes, LLAISYS_MEMCPY_H2D);
    runtime.api()->memcpy_sync(d_in_strides, in_strides, stride_bytes, LLAISYS_MEMCPY_H2D);

    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rearrange_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            d_shape, d_out_strides, d_in_strides, numel, ndim);
        break;
    case LLAISYS_DTYPE_BF16:
        rearrange_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            d_shape, d_out_strides, d_in_strides, numel, ndim);
        break;
    case LLAISYS_DTYPE_F16:
        rearrange_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            d_shape, d_out_strides, d_in_strides, numel, ndim);
        break;
    default:
        runtime.api()->free_device(d_shape);
        runtime.api()->free_device(d_out_strides);
        runtime.api()->free_device(d_in_strides);
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    detail::checkCuda(cudaGetLastError());
    detail::checkCuda(cudaStreamSynchronize(stream));

    runtime.api()->free_device(d_shape);
    runtime.api()->free_device(d_out_strides);
    runtime.api()->free_device(d_in_strides);
}
}
