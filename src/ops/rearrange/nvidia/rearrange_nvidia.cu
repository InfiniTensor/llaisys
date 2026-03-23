#include "rearrange_nvidia.hpp"

#include "../../../device/nvidia/cuda_utils.hpp"

namespace llaisys::ops::nvidia {
namespace {
__global__ void rearrange_kernel(std::byte *out, const std::byte *in, const size_t *shape,
                                 const ptrdiff_t *out_strides, const ptrdiff_t *in_strides, size_t ndim,
                                 size_t elem_size, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    size_t tmp = idx;
    ptrdiff_t out_off = 0;
    ptrdiff_t in_off = 0;
    for (size_t d = 0; d < ndim; ++d) {
        size_t dim = ndim - 1 - d;
        size_t size = shape[dim];
        size_t coord = tmp % size;
        tmp /= size;
        out_off += static_cast<ptrdiff_t>(coord) * out_strides[dim];
        in_off += static_cast<ptrdiff_t>(coord) * in_strides[dim];
    }
    std::byte *dst = out + out_off * static_cast<ptrdiff_t>(elem_size);
    const std::byte *src = in + in_off * static_cast<ptrdiff_t>(elem_size);
    for (size_t i = 0; i < elem_size; ++i) {
        dst[i] = src[i];
    }
}
} // namespace

void rearrange(std::byte *out, const std::byte *in, const size_t *shape, const ptrdiff_t *out_strides,
               const ptrdiff_t *in_strides, size_t ndim, size_t elem_size, size_t numel) {
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    rearrange_kernel<<<blocks, threads>>>(out, in, shape, out_strides, in_strides, ndim, elem_size, numel);
    llaisys::device::nvidia::cuda_check(cudaGetLastError());
}
} // namespace llaisys::ops::nvidia
