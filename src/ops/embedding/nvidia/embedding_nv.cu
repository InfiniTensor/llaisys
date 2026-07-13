#include "../op.hpp"
#include "../../../tensor/tensor.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint> // for int64_t
#include <cstdio>

namespace llaisys::ops::nvidia {

// ==========================================
// Kernel: Embedding
// ==========================================
template<typename T>
__global__ void embedding_kernel(
    T* output,             
    const int64_t* indices, 
    const T* weight,        
    int embedding_dim,     
    size_t n               
) {
    // 1. 计算当前线程负责 Output 中的第几个元素
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 2. 计算当前元素对应的 Token 和维度
        size_t token_idx = idx / embedding_dim;
        
        // dim_idx: 当前处理的是该 Token 向量的第几维 
        size_t dim_idx = idx % embedding_dim;

        // 3. 获取真实的查表索引 (Row ID)
        int64_t row_id = indices[token_idx];

        // 4. 计算 Weight 中的源地址
        size_t weight_offset = row_id * embedding_dim + dim_idx;

        // 5. 搬运数据
        output[idx] = weight[weight_offset];
    }
}

// ==========================================
// Launcher
// ==========================================
template<typename T>
void launch_embedding(tensor_t output, tensor_t index, tensor_t weight) {
    // 1. 获取维度信息
    size_t num_indices = index->numel();
    int embedding_dim = weight->shape().back();     
    
    // Output 总元素个数 = Token数 * 向量维度
    size_t total_elements = num_indices * embedding_dim;

    // 2. 准备指针
    T* d_out = reinterpret_cast<T*>(output->data());
    const int64_t* d_idx = reinterpret_cast<const int64_t*>(index->data());
    const T* d_weight = reinterpret_cast<const T*>(weight->data());

    // 3. 配置 Kernel
    int threads = 256;
    // 向上取整计算 Blocks
    int blocks = (total_elements + threads - 1) / threads;

    embedding_kernel<T><<<blocks, threads>>>(d_out, d_idx, d_weight, embedding_dim, total_elements);
}

// ==========================================
// 入口函数
// ==========================================
void embedding(tensor_t output, tensor_t index, tensor_t weight) {
    auto dtype = weight->dtype();

    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            launch_embedding<float>(output, index, weight);
            break;
        case LLAISYS_DTYPE_F16:
            launch_embedding<__half>(output, index, weight);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_embedding<__nv_bfloat16>(output, index, weight);
            break;
        default:
            fprintf(stderr, "[Embedding NVIDIA] Unsupported DataType: %d\n", dtype);
            abort();
    }
}

} // namespace llaisys::ops::nvidia