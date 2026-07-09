#include "argmax_nvidia.hpp"

#include "../../nvidia_utils.cuh"
#include "../../../core/llaisys_core.hpp"

namespace llaisys::ops::nvidia {
template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    size_t max_i = 0;
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        float max_f = detail::to_float(vals[0]);
        for (size_t i = 1; i < numel; ++i) {
            float cur = detail::to_float(vals[i]);
            if (cur > max_f) {
                max_f = cur;
                max_i = i;
            }
        }
        *max_val = detail::from_float<T>(max_f);
    } else {
        T max_v = vals[0];
        for (size_t i = 1; i < numel; ++i) {
            if (vals[i] > max_v) {
                max_v = vals[i];
                max_i = i;
            }
        }
        *max_val = max_v;
    }
    *max_idx = static_cast<int64_t>(max_i);
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<llaisys::bf16_t *>(max_val),
            reinterpret_cast<const llaisys::bf16_t *>(vals),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<llaisys::fp16_t *>(max_val),
            reinterpret_cast<const llaisys::fp16_t *>(vals),
            numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    detail::checkCuda(cudaGetLastError());
    detail::checkCuda(cudaStreamSynchronize(stream));
}
}
