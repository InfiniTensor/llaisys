#include "linear_nvidia.cuh"

#include "../../nvidia/nvidia_common.cuh"

namespace {

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t dimi, size_t dimj) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    const size_t total = dimi * dimj;
    if (idx >= total) {
        return;
    }

    const size_t col = idx % dimj;
    const float out_v = llaisys::device::nvidia::cuda_utils::toFloat<T>(out[idx]);
    const float bias_v = llaisys::device::nvidia::cuda_utils::toFloat<T>(bias[col]);
    out[idx] = llaisys::device::nvidia::cuda_utils::fromFloat<T>(out_v + bias_v);
}

template <typename T>
void add_bias(std::byte *out, const std::byte *bias, size_t dimi, size_t dimj) {
    constexpr int threads = 256;
    const size_t total = dimi * dimj;
    add_bias_kernel<<<llaisys::ops::nvidia::blocks_for(total, threads), threads, 0, llaisys::ops::nvidia::current_stream()>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(bias),
        dimi,
        dimj);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t dtype,
            size_t dimi,
            size_t dimk,
            size_t dimj) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cudaDataType_t data_type = device::nvidia::cuda_utils::cublasDataType(dtype);
    const cublasComputeType_t compute_type = device::nvidia::cuda_utils::cublasComputeType(dtype);

    LLAISYS_CUBLAS_CHECK(cublasGemmEx(
        current_cublas_handle(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(dimj),
        static_cast<int>(dimi),
        static_cast<int>(dimk),
        &alpha,
        weight,
        data_type,
        static_cast<int>(dimk),
        in,
        data_type,
        static_cast<int>(dimk),
        &beta,
        out,
        data_type,
        static_cast<int>(dimj),
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (bias == nullptr) {
        return;
    }

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return add_bias<float>(out, bias, dimi, dimj);
    case LLAISYS_DTYPE_F16:
        return add_bias<llaisys::fp16_t>(out, bias, dimi, dimj);
    case LLAISYS_DTYPE_BF16:
        return add_bias<llaisys::bf16_t>(out, bias, dimi, dimj);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
