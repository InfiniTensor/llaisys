#include "linear_nvidia.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>   // cuBLAS 库
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace llaisys::ops::nvidia {

// ============================================================================
// 辅助函数：获取或创建 cuBLAS Handle (每个线程一个，简单起见这里每次创建，实际可优化为缓存)
// 注意：在生产环境中，应该将 cublasHandle_t 缓存起来复用，避免重复创建开销。
// 但为了代码简洁和线程安全，这里演示每次调用获取上下文中的 handle 或临时创建。
// 更优解：利用 llaisys 的 context 管理 handle。这里假设我们可以临时创建或使用默认流。
// ============================================================================

// 模板实现：调用 cuBLAS GEMM
template<typename T, typename ComputeType>
void launch_gemm(cublasHandle_t handle, 
                 int m, int n, int k,
                 const T* d_in, const T* d_weight, T* d_out, 
                 const T* d_bias, float alpha, float beta) {
    
    // cuBLAS 默认是列优先 (Column-Major)，而 PyTorch/LLaISYS 通常是行优先 (Row-Major)。
    // 技巧：行优先的 A * B 等价于 列优先的 B^T * A^T。
    // 我们的公式：Out (MxN) = In (MxK) * Weight^T (KxN) + Bias
    // 在 cuBLAS (列优先) 中视为：Out^T (NxM) = Weight (NxK) * In^T (KxM)
    // 所以：
    // A = Weight (N x K), lda = N
    // B = In (K x M), ldb = K
    // C = Out (N x M), ldc = N
    // 运算：C = A * B
    
    const T* A = d_weight; // [N, K]
    const T* B = d_in;     // [K, M] (注意输入被视为 KxM)
    T* C = d_out;          // [N, M]

    int lda = n;       // Weight 的行数
    int ldb = k;       // In 的行数 (逻辑上)
    int ldc = n;       // Out 的行数

    // 选择具体的 GEMM 函数
    if constexpr (std::is_same<T, float>::value) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    n, m, k, 
                    &alpha, 
                    A, lda, 
                    B, ldb, 
                    &beta, 
                    C, ldc);
    } 
    else if constexpr (std::is_same<T, half>::value) {
        // FP16 GEMM
        __half h_alpha = __float2half(alpha);
        __half h_beta = __float2half(beta);
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &h_alpha,
                    (const half*)A, lda,
                    (const half*)B, ldb,
                    &h_beta,
                    (half*)C, ldc);
    }
    else if constexpr (std::is_same<T, nv_bfloat16>::value) {
        // BF16 GEMM (需要 cuBLAS 11.x+)
        __nv_bfloat16 bf_alpha = __float2bfloat16(alpha);
        __nv_bfloat16 bf_beta = __float2bfloat16(beta);
        cublasBfp16Gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        n, m, k,
                        &bf_alpha,
                        (const nv_bfloat16*)A, lda,
                        (const nv_bfloat16*)B, ldb,
                        &bf_beta,
                        (nv_bfloat16*)C, ldc);
    }
}

// ============================================================================
// 对外暴露的 C++ 接口
// ============================================================================
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t n, size_t k) {
    
    // 1. 创建 cuBLAS Handle
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Create failed\n");
        return;
    }

    // 2. 设置流 (可选，使用默认流)
    // cublasSetStream(handle, stream); 

    const float alpha = 1.0f;
    const float beta = 0.0f; // 先设为 0，后面单独加 Bias

    // 3. 执行 GEMM: Out = In * Weight^T
    switch (type) {
        case LLAISYS_DTYPE_F32:
            launch_gemm<float, float>(handle, m, n, k, 
                                      reinterpret_cast<const float*>(in), 
                                      reinterpret_cast<const float*>(weight), 
                                      reinterpret_cast<float*>(out), 
                                      nullptr, alpha, beta);
            break;
        case LLAISYS_DTYPE_F16:
            launch_gemm<half, float>(handle, m, n, k, 
                                     reinterpret_cast<const half*>(in), 
                                     reinterpret_cast<const half*>(weight), 
                                     reinterpret_cast<half*>(out), 
                                     nullptr, alpha, beta);
            break;
        case LLAISYS_DTYPE_BF16:
            launch_gemm<nv_bfloat16, float>(handle, m, n, k, 
                                            reinterpret_cast<const nv_bfloat16*>(in), 
                                            reinterpret_cast<const nv_bfloat16*>(weight), 
                                            reinterpret_cast<nv_bfloat16*>(out), 
                                            nullptr, alpha, beta);
            break;
        default:
            fprintf(stderr, "Unsupported dtype for Linear\n");
            cublasDestroy(handle);
            return;
    }

    // 4. 处理 Bias (如果存在): Out = Out + Bias
    // Bias 是一个向量 [N]，需要广播加到 [M, N] 的每一行
    if (bias != nullptr) {
        // 这里可以写一个简单的 CUDA Kernel 做向量加法，或者复用之前的 add 逻辑
        // 为了简洁，我们内联一个小 Kernel
        auto add_bias_kernel = []__global__ (auto* out_ptr, const auto* bias_ptr, int M, int N) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < M && col < N) {
                out_ptr[row * N + col] += bias_ptr[col];
            }
        };
        
        dim3 block(16, 16);
        dim3 grid((n + 15)/16, (m + 15)/16);

        if (type == LLAISYS_DTYPE_F32) {
            add_bias_kernel<<<grid, block>>>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(bias), m, n);
        } else if (type == LLAISYS_DTYPE_F16) {
            add_bias_kernel<<<grid, block>>>(reinterpret_cast<half*>(out), reinterpret_cast<const half*>(bias), m, n);
        } else if (type == LLAISYS_DTYPE_BF16) {
            add_bias_kernel<<<grid, block>>>(reinterpret_cast<nv_bfloat16*>(out), reinterpret_cast<const nv_bfloat16*>(bias), m, n);
        }
    }

    // 5. 销毁 Handle
    cublasDestroy(handle);
}

} // namespace llaisys::ops::nvidia