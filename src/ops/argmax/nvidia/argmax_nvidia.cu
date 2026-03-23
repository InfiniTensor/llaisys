#include "argmax_nvidia.hpp"

#include "../../../device/nvidia/cuda_utils.hpp"

namespace llaisys::ops::nvidia {
namespace {
template <typename T>
__global__ void argmax_kernel(int64_t *out_idx, T *out_val, const T *vals, size_t numel) {
    float best = llaisys::device::nvidia::ScalarOps<T>::load(vals);
    int64_t best_idx = 0;
    for (size_t i = 1; i < numel; ++i) {
        float v = llaisys::device::nvidia::ScalarOps<T>::load(vals + i);
        if (v > best) {
            best = v;
            best_idx = static_cast<int64_t>(i);
        }
    }
    *out_idx = best_idx;
    llaisys::device::nvidia::ScalarOps<T>::store(out_val, best);
}

template <typename T>
void launch_argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel) {
    argmax_kernel<<<1, 1>>>(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<T *>(max_val),
                            reinterpret_cast<const T *>(vals), numel);
    llaisys::device::nvidia::cuda_check(cudaGetLastError());
}
} // namespace

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_argmax<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return launch_argmax<llaisys::bf16_t>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return launch_argmax<llaisys::fp16_t>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
