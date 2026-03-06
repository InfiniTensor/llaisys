#include "linear_nvidia.hpp"
#include "../../../utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

namespace llaisys::ops::nvidia {

// --- 添加偏置 (Bias) 的 Kernel ---
__global__ void add_bias_kernel_f32(float* out, const float* bias, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        out[idx] += bias[idx % N];
    }
}

__global__ void add_bias_kernel_f16(__half* out, const __half* bias, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        float val = __half2float(out[idx]) + __half2float(bias[idx % N]);
        out[idx] = __float2half(val);
    }
}

__global__ void add_bias_kernel_bf16(void* out, const void* bias, size_t M, size_t N) {
#if __CUDACC_VER_MAJOR__ >= 11
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(out);
        const __nv_bfloat16* bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias);
        
        float val = __bfloat162float(out_ptr[idx]) + __bfloat162float(bias_ptr[idx % N]);
        out_ptr[idx] = __float2bfloat16(val);
    }
#endif
}

// 获取每个线程独享的 cuBLAS 句柄，避免频繁创建销毁带来的巨大开销
cublasHandle_t get_cublas_handle() {
    thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K) {
            
    cublasHandle_t handle = get_cublas_handle();

    // 矩阵乘法的系数: C = alpha * A * B + beta * C
    float alpha_f32 = 1.0f;
    float beta_f32  = 0.0f;

    cudaDataType_t cuda_type;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F; // 统一使用 32F 精度进行中间累加，防止溢出

    switch (type) {
    case LLAISYS_DTYPE_F32:
        cuda_type = CUDA_R_32F;
        break;
    case LLAISYS_DTYPE_F16:
        cuda_type = CUDA_R_16F;
        break;
    case LLAISYS_DTYPE_BF16:
        cuda_type = CUDA_R_16BF;
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    // 调用 Tensor Cores 执行极致速度的矩阵乘法 (利用转置魔法处理行列优先问题)
    // 逻辑等价于: Out(M, N) = In(M, K) @ Weight(N, K)^T
    cublasStatus_t status = cublasGemmEx(
        handle, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha_f32,
        weight, cuda_type, K,
        in,     cuda_type, K,
        &beta_f32,
        out,    cuda_type, N,
        compute_type, CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS Gemm failed! Error code: " + std::to_string(status));
    }

    // 如果有 Bias (偏置)，启动 Kernel 加进去
    if (bias != nullptr) {
        int threads_per_block = 256;
        int blocks_per_grid = (M * N + threads_per_block - 1) / threads_per_block;

        switch (type) {
        case LLAISYS_DTYPE_F32:
            add_bias_kernel_f32<<<blocks_per_grid, threads_per_block>>>(
                reinterpret_cast<float*>(out),
                reinterpret_cast<const float*>(bias),
                M, N
            );
            break;
        case LLAISYS_DTYPE_F16:
            add_bias_kernel_f16<<<blocks_per_grid, threads_per_block>>>(
                reinterpret_cast<__half*>(out),
                reinterpret_cast<const __half*>(bias),
                M, N
            );
            break;
        case LLAISYS_DTYPE_BF16:
            add_bias_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(out, bias, M, N);
            break;
        default:
            break;
        }
    }
}

} // namespace llaisys::ops::nvidia