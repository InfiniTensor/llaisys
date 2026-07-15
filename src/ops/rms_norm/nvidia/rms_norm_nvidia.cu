#include "rms_norm_nvidia.cuh"

#include "../../nvidia/nvidia_common.cuh"

namespace {

int choose_threads(size_t dimj) {
    int threads = 32;
    while (static_cast<size_t>(threads) < dimj && threads < 256) {
        threads <<= 1;
    }
    return threads;
}

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, float eps, size_t dimj) {
    extern __shared__ float shared_sum[];

    const size_t row = static_cast<size_t>(blockIdx.x);
    const size_t tid = static_cast<size_t>(threadIdx.x);
    const T *in_row = in + row * dimj;
    T *out_row = out + row * dimj;

    float local_sum = 0.0f;
    for (size_t col = tid; col < dimj; col += static_cast<size_t>(blockDim.x)) {
        const float value = llaisys::device::nvidia::cuda_utils::toFloat<T>(in_row[col]);
        local_sum += value * value;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(shared_sum[0] / static_cast<float>(dimj) + eps);
    }
    __syncthreads();

    for (size_t col = tid; col < dimj; col += static_cast<size_t>(blockDim.x)) {
        const float value = llaisys::device::nvidia::cuda_utils::toFloat<T>(in_row[col]);
        const float scale = llaisys::device::nvidia::cuda_utils::toFloat<T>(weight[col]);
        out_row[col] = llaisys::device::nvidia::cuda_utils::fromFloat<T>(value * inv_rms * scale);
    }
}

template <typename T>
void rms_norm_impl(std::byte *out,
                   const std::byte *in,
                   const std::byte *weight,
                   float eps,
                   const std::vector<size_t> &shape) {
    const size_t dimi = shape[0];
    const size_t dimj = shape[1];
    const int threads = choose_threads(dimj);
    rms_norm_kernel<<<static_cast<unsigned int>(dimi), threads, static_cast<size_t>(threads) * sizeof(float), llaisys::ops::nvidia::current_stream()>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        eps,
        dimj);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              float eps,
              llaisysDataType_t dtype,
              const std::vector<size_t> &shape) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl<float>(out, in, weight, eps, shape);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl<llaisys::fp16_t>(out, in, weight, eps, shape);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl<llaisys::bf16_t>(out, in, weight, eps, shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
