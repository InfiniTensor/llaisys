#include "rms_norm_nvidia.hpp"

#include "../../nvidia_cuda.cuh"

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, size_t M, size_t d, float eps) {
    size_t row = blockIdx.x;
    if (row >= M) {
        return;
    }

    extern __shared__ float shm[];
    float *s_sum = shm;

    const T *row_in = in + row * d;
    T *row_out = out + row * d;

    float local_sum = 0.0f;
    for (size_t j = threadIdx.x; j < d; j += blockDim.x) {
        float x = to_float(row_in[j]);
        local_sum += x * x;
    }

    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(s_sum[0] / static_cast<float>(d) + eps);
    for (size_t j = threadIdx.x; j < d; j += blockDim.x) {
        float x = to_float(row_in[j]);
        float w = to_float(weight[j]);
        row_out[j] = from_float<T>(x * w * inv_rms);
    }
}

template <typename T>
void launch_rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    size_t M,
    size_t d,
    float eps) {
    const int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
    auto stream = current_stream();
    rms_norm_kernel<<<static_cast<int>(M), threads, shared_bytes, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        M,
        d,
        eps);
    check_cuda(cudaGetLastError(), "rms_norm_kernel launch");
}

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t M,
    size_t d,
    float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_rms_norm<float>(out, in, weight, M, d, eps);
    case LLAISYS_DTYPE_F16:
        return launch_rms_norm<llaisys::fp16_t>(out, in, weight, M, d, eps);
    case LLAISYS_DTYPE_BF16:
        return launch_rms_norm<llaisys::bf16_t>(out, in, weight, M, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
