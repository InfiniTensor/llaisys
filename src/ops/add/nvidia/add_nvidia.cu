#include "add_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>   // half (fp16)
#include <cuda_bf16.h>   // bfloat16
#include <cstdio>

namespace llaisys::ops::nvidia {

// ============================================================================
// CUDA Kernel: 逐元素加法
// 每个线程负责计算一个元素的加法: c[i] = a[i] + b[i]
// ============================================================================
template<typename T> __global__ void add_kernel(T* c,const T* a,const T* b,size_t n){
    size_t i=bolckIdx.x*blockDim.x+threadIdx.x;
    if (i < n) {
        if constexpr (std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value) {
            // 半精度：转 float -> 相加 -> 转回原类型
            float val_a = std::is_same<T, half>::value ? __half2float(a[i]) : __bfloat162float(a[i]);
            float val_b = std::is_same<T, half>::value ? __half2float(b[i]) : __bfloat162float(b[i]);
            float sum = val_a + val_b;
            
            if constexpr (std::is_same<T, half>::value) {
                c[i] = __float2half(sum);
            } else {
                c[i] = __float2bfloat16(sum);
            }
        } else {
            // float32：直接相加
            c[i] = a[i] + b[i];
        }
    }
}

// ============================================================================
// 模板分发函数：根据类型启动 Kernel
// ============================================================================
template<typename T>
void launch_add(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    const T* d_a = reinterpret_cast<const T*>(a);
    const T* d_b = reinterpret_cast<const T*>(b);
    T* d_c = reinterpret_cast<T*>(c);

    // 配置参数
    const int threads_per_block = 256;
    // 计算需要的 Block 数量 (向上取整)
    const int num_blocks = (numel + threads_per_block - 1) / threads_per_block;

    // 启动 Kernel
    add_kernel<T><<<num_blocks, threads_per_block>>>(d_c, d_a, d_b, numel);

    // 检查启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Add Kernel Launch failed: %s\n", cudaGetErrorString(err));
    }
    
    // 注意：这里不需要同步 (cudaDeviceSynchronize)
    // 上层框架会在需要结果时自动同步，或者通过 Stream 异步执行
}

// ============================================================================
// 对外暴露的 C++ 接口
// ============================================================================
void add(std::byte *c, const std::byte *a, const std::byte *b, 
         llaisysDataType_t type, size_t numel) {
    
    switch (type) {
        case LLAISYS_DTYPE_F32:
            launch_add<float>(c, a, b, numel);
            break;
        case LLAISYS_DTYPE_F16:
            launch_add<half>(c, a, b, numel);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_add<nv_bfloat16>(c, a, b, numel);
            break;
        default:
            fprintf(stderr, "Unsupported dtype for NVIDIA Add: %d\n", type);
            break;
    }
}

} // namespace llaisys::ops::nvidia