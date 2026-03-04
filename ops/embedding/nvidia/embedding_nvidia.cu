#include "embedding_nvidia.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

namespace llaisys::ops::nvidia {

// ============================================================================
// CUDA Kernel: Embedding Lookup
// 每个线程负责处理一个索引 (即输出矩阵的一行)
// ============================================================================
template<typename T>
__global__ void embedding_kernel(T* out, const int64_t* indices, const T* weight, 
                                 size_t num_indices, size_t embd_dim, size_t weight_rows) {
    // 当前线程处理的索引位置 (对应输出矩阵的第几行)
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < num_indices) {
        // 1. 读取索引值
        int64_t table_idx = indices[row_idx];

        // 2. 边界检查 
        if (table_idx < 0 || static_cast<size_t>(table_idx) >= weight_rows) {
            if (threadIdx.x == 0) {
                printf("Embedding Error: Index %ld out of range [0, %lu)\n", table_idx, weight_rows);
            }
            return;
        }

        // 3. 计算源地址和目标地址
        // 源：权重表中第 table_idx 行
        const T* src_row = weight + table_idx * embd_dim;
        // 目标：输出矩阵中第 row_idx 行
        T* dst_row = out + row_idx * embd_dim;

        // 4. 逐元素拷贝 (也可以用 float4 优化，但这里先保持简单清晰)
        for (size_t i = 0; i < embd_dim; ++i) {
            dst_row[i] = src_row[i];
        }
    }
}

// ============================================================================
// 模板分发函数
// ============================================================================
template<typename T>
void launch_embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
                      size_t num_indices, size_t embd_dim, size_t weight_rows) {
    const T* d_weight = reinterpret_cast<const T*>(weight);
    const int64_t* d_indices = reinterpret_cast<const int64_t*>(index);
    T* d_out = reinterpret_cast<T*>(out);

    // 配置参数
    const int threads_per_block = 256;
    // 需要的 Block 数量 = 索引个数 / 256
    const int num_blocks = (num_indices + threads_per_block - 1) / threads_per_block;

    // 启动 Kernel
    embedding_kernel<T><<<num_blocks, threads_per_block>>>(
        d_out, d_indices, d_weight, num_indices, embd_dim, weight_rows
    );

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Embedding Kernel Launch failed: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// 对外暴露的 C++ 接口
// ============================================================================
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t type, size_t num_indices, size_t embd_dim, size_t weight_rows) {
    
    switch (type) {
        case LLAISYS_DTYPE_F32:
            launch_embedding<float>(out, index, weight, num_indices, embd_dim, weight_rows);
            break;
        case LLAISYS_DTYPE_F16:
            launch_embedding<half>(out, index, weight, num_indices, embd_dim, weight_rows);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_embedding<nv_bfloat16>(out, index, weight, num_indices, embd_dim, weight_rows);
            break;
        default:
            fprintf(stderr, "Unsupported dtype for NVIDIA Embedding: %d\n", type);
            break;
    }
}

} // namespace llaisys::ops::nvidia