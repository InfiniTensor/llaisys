#include "embedding_nvidia.hpp"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

namespace llaisys::ops::nvidia {

// --- F32 Kernel ---
// 🚨 修改点：index 指针类型改为 const int64_t*
__global__ void embedding_kernel_f32(float* out, const int64_t* index, const float* weight, size_t num_indices, size_t vocab_size, size_t embedding_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices * embedding_dim) {
        size_t token_idx = idx / embedding_dim;
        size_t dim_idx = idx % embedding_dim;
        
        int64_t word_id = index[token_idx]; // 读取 64 位整型
        
        // 增加 >= 0 的越界保护，因为有符号整型可能是负数
        if (word_id >= 0 && word_id < vocab_size) {
            out[idx] = weight[word_id * embedding_dim + dim_idx];
        } else {
            out[idx] = 0.0f;
        }
    }
}

// --- F16 Kernel ---
__global__ void embedding_kernel_f16(void* out, const int64_t* index, const void* weight, size_t num_indices, size_t vocab_size, size_t embedding_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices * embedding_dim) {
        size_t token_idx = idx / embedding_dim;
        size_t dim_idx = idx % embedding_dim;
        int64_t word_id = index[token_idx];
        
        if (word_id >= 0 && word_id < vocab_size) {
            reinterpret_cast<__half*>(out)[idx] = reinterpret_cast<const __half*>(weight)[word_id * embedding_dim + dim_idx];
        } else {
            reinterpret_cast<__half*>(out)[idx] = __float2half(0.0f);
        }
    }
}

// --- BF16 Kernel ---
__global__ void embedding_kernel_bf16(void* out, const int64_t* index, const void* weight, size_t num_indices, size_t vocab_size, size_t embedding_dim) {
#if __CUDACC_VER_MAJOR__ >= 11
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices * embedding_dim) {
        size_t token_idx = idx / embedding_dim;
        size_t dim_idx = idx % embedding_dim;
        int64_t word_id = index[token_idx];
        
        if (word_id >= 0 && word_id < vocab_size) {
            reinterpret_cast<__nv_bfloat16*>(out)[idx] = reinterpret_cast<const __nv_bfloat16*>(weight)[word_id * embedding_dim + dim_idx];
        } else {
            reinterpret_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16(0.0f);
        }
    }
#endif
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t type, size_t num_indices, size_t vocab_size, size_t embedding_dim) {
               
    size_t total_elements = num_indices * embedding_dim;
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    // 🚨 修正强转：将传入的 index 解释为 int64_t 指针
    const int64_t* index_ptr = reinterpret_cast<const int64_t*>(index);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel_f32<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<float*>(out), index_ptr, reinterpret_cast<const float*>(weight), 
            num_indices, vocab_size, embedding_dim
        );
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel_f16<<<blocks_per_grid, threads_per_block>>>(
            out, index_ptr, weight, num_indices, vocab_size, embedding_dim
        );
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(
            out, index_ptr, weight, num_indices, vocab_size, embedding_dim
        );
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia