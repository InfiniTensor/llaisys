#include "rms_norm_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <algorithm>
#include <cmath>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void rmsNormKernel(T *out, const T *input, const T *weight, size_t hidden_size, float eps) {
    __shared__ float shared_sum[256];

    const size_t row = static_cast<size_t>(blockIdx.x);
    const size_t row_offset = row * hidden_size;
    const int tid = threadIdx.x;

    float sum_sq = 0.0f;
    for (size_t col = static_cast<size_t>(tid); col < hidden_size; col += blockDim.x) {
        const float value = llaisys::device::nvidia::scalarToFloat<T>(input[row_offset + col]);
        sum_sq += value * value;
    }

    shared_sum[tid] = sum_sq;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(shared_sum[0] / static_cast<float>(hidden_size) + eps);
    for (size_t col = static_cast<size_t>(tid); col < hidden_size; col += blockDim.x) {
        const float value = llaisys::device::nvidia::scalarToFloat<T>(input[row_offset + col]);
        const float scale = llaisys::device::nvidia::scalarToFloat<T>(weight[col]);
        out[row_offset + col] = llaisys::device::nvidia::floatToScalar<T>(value * inv_rms * scale);
    }
}

} // namespace

void rms_norm(std::byte *out, const std::byte *input, const std::byte *weight, llaisysDataType_t type, size_t batch,
              size_t hidden_size, float eps, llaisysStream_t stream) {
    if (batch == 0 || hidden_size == 0) {
        return;
    }

    constexpr int threads = 256;
    const cudaStream_t cuda_stream = llaisys::device::nvidia::toCudaStream(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rmsNormKernel<<<static_cast<int>(batch), threads, 0, cuda_stream>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(input),
            reinterpret_cast<const float *>(weight),
            hidden_size,
            eps);
        break;
    case LLAISYS_DTYPE_F16:
        rmsNormKernel<<<static_cast<int>(batch), threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(input),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            hidden_size,
            eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rmsNormKernel<<<static_cast<int>(batch), threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(input),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            hidden_size,
            eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
