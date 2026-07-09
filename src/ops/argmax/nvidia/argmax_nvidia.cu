#include "argmax_nvidia.hpp"

#include "../../nvidia_cuda.cuh"

#include <math_constants.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    constexpr int kBlockSize = 256;
    __shared__ float s_val[kBlockSize];
    __shared__ int64_t s_idx[kBlockSize];

    int tid = threadIdx.x;

    float local_val = -CUDART_INF_F;
    int64_t local_idx = 0;
    bool has = false;

    for (size_t i = static_cast<size_t>(tid); i < numel; i += blockDim.x) {
        float v = to_float(vals[i]);
        if (!has || v > local_val || (v == local_val && static_cast<int64_t>(i) < local_idx)) {
            local_val = v;
            local_idx = static_cast<int64_t>(i);
            has = true;
        }
    }

    s_val[tid] = local_val;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float rhs_val = s_val[tid + stride];
            int64_t rhs_idx = s_idx[tid + stride];
            if (rhs_val > s_val[tid] || (rhs_val == s_val[tid] && rhs_idx < s_idx[tid])) {
                s_val[tid] = rhs_val;
                s_idx[tid] = rhs_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_idx = s_idx[0];
        *max_val = from_float<T>(s_val[0]);
    }
}

template <typename T>
void launch_argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel) {
    constexpr int kBlockSize = 256;
    auto stream = current_stream();
    argmax_kernel<<<1, kBlockSize, 0, stream>>>(
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        numel);
    check_cuda(cudaGetLastError(), "argmax_kernel launch");
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_argmax<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return launch_argmax<llaisys::fp16_t>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return launch_argmax<llaisys::bf16_t>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
