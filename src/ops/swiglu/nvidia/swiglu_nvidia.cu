#include "swiglu_nvidia.cuh"

#include "../../nvidia/nvidia_common.cuh"

namespace {

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= numel) {
        return;
    }

    const float g = llaisys::device::nvidia::cuda_utils::toFloat<T>(gate[idx]);
    const float u = llaisys::device::nvidia::cuda_utils::toFloat<T>(up[idx]);
    const float silu = g / (1.0f + expf(-g));
    out[idx] = llaisys::device::nvidia::cuda_utils::fromFloat<T>(u * silu);
}

template <typename T>
void swiglu_impl(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel) {
    constexpr int threads = 256;
    swiglu_kernel<<<llaisys::ops::nvidia::blocks_for(numel, threads), threads, 0, llaisys::ops::nvidia::current_stream()>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl<llaisys::fp16_t>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl<llaisys::bf16_t>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
