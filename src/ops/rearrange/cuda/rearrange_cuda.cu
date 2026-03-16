#include "rearrange_cuda.cuh"
#include "../../cuda_utils.cuh"

#include <cstring>

// Max supported dimensions for device-side arrays
#define MAX_DIMS 8

__global__ void rearrange_kernel(void *out, const void *in,
                                 const size_t *d_shape,
                                 const ptrdiff_t *d_out_strides,
                                 const ptrdiff_t *d_in_strides,
                                 size_t ndim, size_t esize, size_t numel) {
    size_t flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= numel) return;

    // Convert flat index to multi-dimensional index
    size_t remaining = flat_idx;
    ptrdiff_t src_off = 0;
    ptrdiff_t dst_off = 0;
    for (size_t d = 0; d < ndim; d++) {
        size_t prod = 1;
        for (size_t dd = d + 1; dd < ndim; dd++) prod *= d_shape[dd];
        size_t coord = remaining / prod;
        remaining %= prod;
        src_off += coord * d_in_strides[d];
        dst_off += coord * d_out_strides[d];
    }

    const char *src = reinterpret_cast<const char *>(in) + src_off * esize;
    char *dst = reinterpret_cast<char *>(out) + dst_off * esize;
    for (size_t b = 0; b < esize; b++) {
        dst[b] = src[b];
    }
}

namespace llaisys::ops::cuda {
void rearrange(std::byte *out, const std::byte *in,
               const size_t *shape, const ptrdiff_t *out_strides, const ptrdiff_t *in_strides,
               size_t ndim, size_t esize, size_t numel) {
    // Copy shape and strides to device
    size_t *d_shape;
    ptrdiff_t *d_out_strides, *d_in_strides;
    cudaMalloc(&d_shape, ndim * sizeof(size_t));
    cudaMalloc(&d_out_strides, ndim * sizeof(ptrdiff_t));
    cudaMalloc(&d_in_strides, ndim * sizeof(ptrdiff_t));
    cudaMemcpy(d_shape, shape, ndim * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides, ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice);

    rearrange_kernel<<<cuda_grid_size(numel), CUDA_BLOCK_SIZE>>>(
        out, in, d_shape, d_out_strides, d_in_strides, ndim, esize, numel);
    CUDA_KERNEL_CHECK();

    cudaFree(d_shape);
    cudaFree(d_out_strides);
    cudaFree(d_in_strides);
}
} // namespace llaisys::ops::cuda
