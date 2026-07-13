#include "../op.hpp"
#include "../../../tensor/tensor.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdio>

namespace llaisys::ops::nvidia {

template<typename T>
struct Pair {
    T val;
    int64_t idx;
}; 

// argmax 核函数：支持同时输出 index 和 value
template<typename T>
__global__ void argmax_kernel(int64_t* idx_out, T* val_out, const T* input, int n_cols) {
    // 1. 确定行号和线程号
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 2. 确定行首地址
    const T* row_ptr = input + row * n_cols;

    // 初始化局部最大值
    T max_val = -1e20f; // 简单粗暴的极小值
    int64_t max_idx = -1;

    // 3. 遍历该行所有元素，计算最大值和索引 (Grid-Stride Loop)
    for (int i = tid; i < n_cols; i += blockDim.x) {
        T val = row_ptr[i];
        // 强制转 float 比较，确保半精度类型的兼容性
        if ((float)val > (float)max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    // 4. 将每个线程的最大值和索引存入共享内存
    extern __shared__ char s_mem[];
    Pair<T>* s_data = reinterpret_cast<Pair<T>*>(s_mem);
    s_data[tid] = {max_val, max_idx};
    __syncthreads();

    // 5. 归约计算最终的最大值和索引
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // 比较并交换
            if ((float)s_data[tid + s].val > (float)s_data[tid].val) {
                s_data[tid] = s_data[tid + s];
            }
        }
        __syncthreads();
    }

    // 6. 将结果写入输出 (0号线程负责)
    if (tid == 0) {
        // 写入索引
        if (idx_out != nullptr) {
            idx_out[row] = s_data[0].idx;
        }
        if (val_out != nullptr) {
            val_out[row] = s_data[0].val;
        }
    }
}

// Launcher：接收三个 Tensor
template<typename T>
void launch_argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    int n_cols = vals->shape().back(); 
    int n_rows = vals->numel() / n_cols;

    // 获取数据指针
    int64_t* d_idx = (max_idx) ? reinterpret_cast<int64_t*>(max_idx->data()) : nullptr;
    T* d_val   = (max_val) ? reinterpret_cast<T*>(max_val->data()) : nullptr;
    const T* d_in = reinterpret_cast<const T*>(vals->data());

    int threads = 256;
    int blocks = n_rows;

    size_t shared_mem_size = threads * sizeof(Pair<T>);
    
    // 传入两个输出指针
    argmax_kernel<T><<<blocks, threads, shared_mem_size>>>(d_idx, d_val, d_in, n_cols);
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 根据输入数据的类型进行分发
    auto dtype = vals->dtype();

    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            launch_argmax<float>(max_idx, max_val, vals);
            break;
        case LLAISYS_DTYPE_F16:
            launch_argmax<__half>(max_idx, max_val, vals);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_argmax<__nv_bfloat16>(max_idx, max_val, vals);
            break;
        default:
            fprintf(stderr, "[Argmax] Unsupported DataType: %d\n", dtype);
            abort();
    }
}

} // namespace llaisys::ops::nvidia