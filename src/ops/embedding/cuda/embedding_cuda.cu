#include "embedding_cuda.cuh"
#include "../../cuda_utils.cuh"

__global__ void embedding_kernel(void *out, const int64_t *index, const void *weight,
                                 size_t esize, size_t n_idx, size_t embd_dim) {
    size_t i = blockIdx.x;
    size_t j = threadIdx.x + blockIdx.y * blockDim.x;
    if (i >= n_idx || j >= embd_dim) return;

    int64_t row = index[i];
    size_t src_off = row * embd_dim * esize + j * esize;
    size_t dst_off = i * embd_dim * esize + j * esize;

    const char *src = reinterpret_cast<const char *>(weight) + src_off;
    char *dst = reinterpret_cast<char *>(out) + dst_off;

    for (size_t b = 0; b < esize; b++) {
        dst[b] = src[b];
    }
}

namespace llaisys::ops::cuda {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t n_idx, size_t embd_dim) {
    size_t esize = cuda_dsize(dtype);
    int threads_per_block = 256;
    dim3 grid(n_idx, (embd_dim + threads_per_block - 1) / threads_per_block);
    dim3 block(threads_per_block);
    embedding_kernel<<<grid, block>>>(out, reinterpret_cast<const int64_t *>(index),
                                      weight, esize, n_idx, embd_dim);
    CUDA_KERNEL_CHECK();
}
} // namespace llaisys::ops::cuda
