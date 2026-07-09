#include "linear_cuda.cuh"

#include "../../../device/nvidia/nvidia_resource.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <sstream>
#include <stdexcept>

#define CUBLAS_CHECK(call)                                   \
    do {                                                     \
        cublasStatus_t _st = (call);                         \
        if (_st != CUBLAS_STATUS_SUCCESS) {                  \
            std::ostringstream _oss;                         \
            _oss << "cuBLAS error " << static_cast<int>(_st) \
                 << " at " << __FILE__ << ":" << __LINE__;   \
            throw std::runtime_error(_oss.str());            \
        }                                                    \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Bias add kernel（cuBLAS GEMM 不原生支持 bias，用独立 kernel 追加）
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
__global__ void add_bias_kernel(T *__restrict__ out,
                                const T *__restrict__ bias,
                                size_t batch_size,
                                size_t out_features) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) {
        return;
    }

    size_t col = idx % out_features;
    float o_val, b_val;
    if constexpr (std::is_same_v<T, __half>) {
        o_val = __half2float(out[idx]);
        b_val = __half2float(bias[col]);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        o_val = __bfloat162float(out[idx]);
        b_val = __bfloat162float(bias[col]);
    } else {
        o_val = static_cast<float>(out[idx]);
        b_val = static_cast<float>(bias[col]);
    }
    float result = o_val + b_val;
    if constexpr (std::is_same_v<T, __half>) {
        out[idx] = __float2half(result);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        out[idx] = __float2bfloat16(result);
    } else {
        out[idx] = static_cast<T>(result);
    }
}

namespace llaisys::ops::nvidia {

// ─────────────────────────────────────────────────────────────────────────────
// 行主序 Y = X * W^T，等价于列主序 Y^T = W * X^T
// 调用公式（cuBLAS col-major）：
//   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &a, W, K, X, K, &b, Y, N)
// 其中 M=batch_size, K=in_features, N=out_features
// ─────────────────────────────────────────────────────────────────────────────
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t type,
            size_t batch_size, size_t in_features, size_t out_features) {
    cublasHandle_t handle = llaisys::device::nvidia::getCublasHandle();

    int M = static_cast<int>(batch_size);
    int K = static_cast<int>(in_features);
    int N = static_cast<int>(out_features);

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        // TF32 Tensor Core（默认由 CUBLAS_TF32_TENSOR_OP_MATH 开启）
        const float alpha = 1.0f, beta_val = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            reinterpret_cast<const float *>(weight), K,
            reinterpret_cast<const float *>(in), K,
            &beta_val,
            reinterpret_cast<float *>(out), N));

        if (bias) {
            size_t numel = batch_size * out_features;
            int grid = static_cast<int>((numel + 255) / 256);
            add_bias_kernel<float><<<grid, 256>>>(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(bias),
                batch_size, out_features);
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        // FP16 Tensor Core（A100 支持 CUBLAS_COMPUTE_16F）
        const __half alpha = __float2half(1.0f);
        const __half beta_val = __float2half(0.0f);
        CUBLAS_CHECK(cublasHgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            reinterpret_cast<const __half *>(weight), K,
            reinterpret_cast<const __half *>(in), K,
            &beta_val,
            reinterpret_cast<__half *>(out), N));

        if (bias) {
            size_t numel = batch_size * out_features;
            int grid = static_cast<int>((numel + 255) / 256);
            add_bias_kernel<__half><<<grid, 256>>>(
                reinterpret_cast<__half *>(out),
                reinterpret_cast<const __half *>(bias),
                batch_size, out_features);
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        // BF16 Tensor Core：A100 SM_80 峰值 312 TFLOPS
        // 使用 cublasGemmEx 指定 BF16 计算类型
        const float alpha = 1.0f, beta_val = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            weight, CUDA_R_16BF, K,
            in, CUDA_R_16BF, K,
            &beta_val,
            out, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F, // 累加器使用 f32 确保精度
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        if (bias) {
            size_t numel = batch_size * out_features;
            int grid = static_cast<int>((numel + 255) / 256);
            add_bias_kernel<__nv_bfloat16><<<grid, 256>>>(
                reinterpret_cast<__nv_bfloat16 *>(out),
                reinterpret_cast<const __nv_bfloat16 *>(bias),
                batch_size, out_features);
        }
        break;
    }
    default:
        throw std::runtime_error("linear CUDA: unsupported data type");
    }
}

} // namespace llaisys::ops::nvidia
