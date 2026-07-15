#include "add_nvidia.cuh"

#include "../../nvidia/nvidia_common.cuh"

namespace {

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= numel) {
        return;
    }

    const float lhs = llaisys::device::nvidia::cuda_utils::toFloat<T>(a[idx]);
    const float rhs = llaisys::device::nvidia::cuda_utils::toFloat<T>(b[idx]);
    c[idx] = llaisys::device::nvidia::cuda_utils::fromFloat<T>(lhs + rhs);
}

template <typename T>
void add_impl(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    constexpr int threads = 256;
    add_kernel<<<llaisys::ops::nvidia::blocks_for(numel, threads), threads, 0, llaisys::ops::nvidia::current_stream()>>>(
        reinterpret_cast<T *>(c),
        reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b),
        numel);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return add_impl<float>(c, a, b, numel);
    case LLAISYS_DTYPE_F16:
        return add_impl<llaisys::fp16_t>(c, a, b, numel);
    case LLAISYS_DTYPE_BF16:
        return add_impl<llaisys::bf16_t>(c, a, b, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
