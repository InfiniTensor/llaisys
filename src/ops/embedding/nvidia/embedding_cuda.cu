#include "embedding_cuda.cuh"

#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Embedding lookup kernel
// gridDim.x  = num_indices
// blockDim.x = min(embedding_dim, 1024)
// 每个 block 负责将 weight[index[i], :] 复制到 out[i, :]
// 使用 float4 宽加载（16 bytes / 线程）提升 DRAM 带宽
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
__global__ void embedding_kernel(T *__restrict__ out,
                                 const int64_t *__restrict__ index,
                                 const T *__restrict__ weight,
                                 size_t embedding_dim) {
    size_t row = blockIdx.x;
    int64_t idx = index[row];
    const T *src = weight + idx * embedding_dim;
    T *dst = out + row * embedding_dim;

    for (size_t i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// float4 特化版本（embedding_dim 为 4 的倍数时使用）
__global__ void embedding_f32_kernel(float *__restrict__ out,
                                     const int64_t *__restrict__ index,
                                     const float *__restrict__ weight,
                                     size_t dim4, size_t embedding_dim) {
    size_t row = blockIdx.x;
    int64_t idx = index[row];
    const float4 *src4 = reinterpret_cast<const float4 *>(weight + idx * embedding_dim);
    float4 *dst4 = reinterpret_cast<float4 *>(out + row * embedding_dim);

    for (size_t i = threadIdx.x; i < dim4; i += blockDim.x) {
        dst4[i] = src4[i];
    }
    // 尾部处理
    const float *src = weight + idx * embedding_dim;
    float *dst = out + row * embedding_dim;
    size_t tail_start = dim4 * 4;
    for (size_t i = tail_start + threadIdx.x; i < embedding_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

namespace llaisys::ops::nvidia {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t num_indices, size_t embedding_dim) {
    if (num_indices == 0 || embedding_dim == 0) {
        return;
    }

    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);
    const int BLOCK = static_cast<int>(embedding_dim < 1024 ? embedding_dim : 1024);
    const int GRID = static_cast<int>(num_indices);

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        size_t dim4 = embedding_dim / 4;
        if (dim4 > 0 && embedding_dim % 4 == 0) {
            int blk = static_cast<int>(dim4 < 256 ? dim4 : 256);
            embedding_f32_kernel<<<GRID, blk>>>(
                reinterpret_cast<float *>(out),
                idx_ptr,
                reinterpret_cast<const float *>(weight),
                dim4, embedding_dim);
        } else {
            embedding_kernel<float><<<GRID, BLOCK>>>(
                reinterpret_cast<float *>(out),
                idx_ptr,
                reinterpret_cast<const float *>(weight),
                embedding_dim);
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        // __half 与 uint16_t 等宽，可安全 reinterpret
        embedding_kernel<uint16_t><<<GRID, BLOCK>>>(
            reinterpret_cast<uint16_t *>(out),
            idx_ptr,
            reinterpret_cast<const uint16_t *>(weight),
            embedding_dim);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        embedding_kernel<uint16_t><<<GRID, BLOCK>>>(
            reinterpret_cast<uint16_t *>(out),
            idx_ptr,
            reinterpret_cast<const uint16_t *>(weight),
            embedding_dim);
        break;
    }
    default:
        throw std::runtime_error("embedding CUDA: unsupported data type");
    }
}

} // namespace llaisys::ops::nvidia
