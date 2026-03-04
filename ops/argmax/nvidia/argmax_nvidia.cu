#include "argmax_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>   // half (fp16)
#include <cuda_bf16.h>   // bfloat16
#include <cstdio>
#include <limits>
#include <cstdint>

namespace llaisys::ops::nvidia {

// ============================================================================
// 辅助结构体：用于在归约过程中同时保存值 (float) 和索引 (int64_t)
// ============================================================================
struct ValIdx {
    float val;
    int64_t idx;
};

// ============================================================================
// Kernel 1: 块级归约
// 每个 Block 负责处理一段数据，找出该段的最大值 (val, idx)，写入 block_results
// ============================================================================
template<typename T>
__global__ void argmax_block_kernel(const T* input, ValIdx* block_results, size_t n) {
    // 共享内存：用于块内归约，大小由启动参数决定
    extern __shared__ ValIdx sdata[];

    unsigned int tid = threadIdx.x;
    // 计算全局索引
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // 初始化当前线程的值
    float my_val = -std::numeric_limits<float>::infinity();
    int64_t my_idx = -1;

    if (i < n) {
        // 读取数据并转换为 float 进行比较
        if constexpr (std::is_same<T, half>::value) {
            my_val = __half2float(input[i]);
        } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
            my_val = __bfloat162float(input[i]);
        } else {
            // float32 直接转换
            my_val = static_cast<float>(input[i]);
        }
        my_idx = static_cast<int64_t>(i);
    }

    // 写入共享内存
    sdata[tid] = {my_val, my_idx};
    __syncthreads();

    // 块内归约：树形归约算法
    // 步长从 blockSize/2 开始，每次减半
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s].val > sdata[tid].val) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    // 每个块的第一个线程将局部结果写入全局数组
    if (tid == 0) {
        block_results[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Kernel 2: 全局归约
// 对 block_results (数量很少，等于 Block 的数量) 再做一次 Argmax
// ============================================================================
__global__ void argmax_final_kernel(const ValIdx* block_results, float* final_val, int64_t* final_idx, int block_count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float max_v = -std::numeric_limits<float>::infinity();
        int64_t max_i = -1;

        for (int i = 0; i < block_count; ++i) {
            if (block_results[i].val > max_v) {
                max_v = block_results[i].val;
                max_i = block_results[i].idx;
            }
        }
        *final_val = max_v;
        *final_idx = max_i;
    }
}

// ============================================================================
// 模板分发函数：根据类型启动 Kernel
// ============================================================================
template<typename T>
void launch_argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel) {
    const T* d_in = reinterpret_cast<const T*>(vals);
    
    // 设备指针用于临时存储
    float* d_final_val = nullptr;
    int64_t* d_final_idx = nullptr;
    ValIdx* d_block_results = nullptr;

    // 配置参数
    const int threads_per_block = 256;
    // 计算需要多少个 Block
    int blocks = (numel + threads_per_block - 1) / threads_per_block;
    if (blocks == 0) blocks = 1; 
    // 限制最大 Block 数，防止显存分配过大（虽然 256 线程/block 通常不会太大）
    if (blocks > 1024) blocks = 1024; 

    // 1. 分配临时内存
    cudaError_t err;
    err = cudaMalloc(&d_block_results, blocks * sizeof(ValIdx));
    if (err != cudaSuccess) {
        fprintf(stderr, "Argmax CUDA Malloc block_results failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&d_final_val, sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Argmax CUDA Malloc final_val failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_block_results); return;
    }

    err = cudaMalloc(&d_final_idx, sizeof(int64_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Argmax CUDA Malloc final_idx failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_block_results); cudaFree(d_final_val); return;
    }

    // 2. 启动第一个 Kernel (块级归约)
    // 共享内存大小 = threads_per_block * sizeof(ValIdx)
    argmax_block_kernel<T><<<blocks, threads_per_block, threads_per_block * sizeof(ValIdx)>>>(
        d_in, d_block_results, numel
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Argmax Kernel 1 Launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_block_results); cudaFree(d_final_val); cudaFree(d_final_idx);
        return;
    }

    // 3. 启动第二个 Kernel (全局归约)
    argmax_final_kernel<<<1, 1>>>(d_block_results, d_final_val, d_final_idx, blocks);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Argmax Kernel 2 Launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_block_results); cudaFree(d_final_val); cudaFree(d_final_idx);
        return;
    }

    // 4. 将结果拷贝回主机 (Host)
    float h_final_val = 0.0f;
    int64_t h_final_idx = 0;

    cudaMemcpy(&h_final_val, d_final_val, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_final_idx, d_final_idx, sizeof(int64_t), cudaMemcpyDeviceToHost);

    // 5. 写回输出指针
    int64_t* h_idx_ptr = reinterpret_cast<int64_t*>(max_idx);
    *h_idx_ptr = h_final_idx;

    T* h_val_ptr = reinterpret_cast<T*>(max_val);
    if constexpr (std::is_same<T, half>::value) {
        *h_val_ptr = __float2half(h_final_val);
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
        *h_val_ptr = __float2bfloat16(h_final_val);
    } else {
        *h_val_ptr = static_cast<T>(h_final_val);
    }

    // 6. 清理临时内存
    cudaFree(d_block_results);
    cudaFree(d_final_val);
    cudaFree(d_final_idx);
}

// ============================================================================
// 对外暴露的 C++ 接口
// ============================================================================
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, 
            llaisysDataType_t type, size_t numel) {
    
    switch (type) {
        case LLAISYS_DTYPE_F32:
            launch_argmax<float>(max_idx, max_val, vals, numel);
            break;
        case LLAISYS_DTYPE_F16:
            launch_argmax<half>(max_idx, max_val, vals, numel);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_argmax<nv_bfloat16>(max_idx, max_val, vals, numel);
            break;
        default:
            fprintf(stderr, "Unsupported dtype for NVIDIA Argmax: %d\n", type);
            break;
    }
}

} // namespace llaisys::ops::nvidia