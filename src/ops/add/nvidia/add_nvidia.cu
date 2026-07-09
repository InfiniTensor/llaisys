#include "add_nvidia.hpp"

#include "../../nvidia_utils.cuh"
#include "../../../core/llaisys_core.hpp"

namespace llaisys::ops::nvidia {
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        float av = detail::to_float(a[idx]);
        float bv = detail::to_float(b[idx]);
        c[idx] = detail::from_float<T>(av + bv);
    } else {
        c[idx] = a[idx] + b[idx];
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(c),
            reinterpret_cast<const llaisys::bf16_t *>(a),
            reinterpret_cast<const llaisys::bf16_t *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(c),
            reinterpret_cast<const llaisys::fp16_t *>(a),
            reinterpret_cast<const llaisys::fp16_t *>(b),
            numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    detail::checkCuda(cudaGetLastError());
    detail::checkCuda(cudaStreamSynchronize(stream));
}
}
