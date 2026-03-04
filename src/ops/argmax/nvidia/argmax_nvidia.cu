#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"
#include "argmax_nvidia.cuh"
#include <cstdint>

namespace {

template <typename T>
__device__ __forceinline__ void warp_argmax(T local_val, int64_t local_idx, T &max_val, int64_t &max_idx) {
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        T other_val = __shfl_down_sync(0xffffffff, local_val, stride);
        int64_t other_idx = __shfl_down_sync(0xffffffff, local_idx, stride);

        if (other_val > local_val || (other_val == local_val && other_idx < local_idx)) {
            local_val = other_val;
            local_idx = other_idx;
        }
    }

    if (threadIdx.x % 32 == 0) {
        max_val = local_val;
        max_idx = local_idx;
    }
}

template <typename T, const int BLOCK_SIZE>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    constexpr int warp_per_block = BLOCK_SIZE / 32;

    int tid = threadIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    __shared__ T vals_shared[warp_per_block];
    __shared__ int64_t idxs_shared[warp_per_block];

    // 0. 线程级别求局部最大值
    T thread_max_val = static_cast<T>(-INFINITY);
    int64_t thread_max_idx = -1;
    for (int i = tid; i < numel; i += blockDim.x) {
        T local_val = vals[i];
        if (local_val > thread_max_val || (local_val == thread_max_val && i < thread_max_idx)) {
            thread_max_val = local_val;
            thread_max_idx = i;
        }
    }

    // 1.warp内规约
    T warp_max_val = thread_max_val;
    int64_t warp_max_idx = thread_max_idx;
    warp_argmax(thread_max_val, thread_max_idx, warp_max_val, warp_max_idx);

    if (lane_id == 0) {
        vals_shared[warp_id] = warp_max_val;
        idxs_shared[warp_id] = warp_max_idx;
    }
    __syncthreads();

    // 2. 用 warp 0 对共享内存里的各 warp 结果做规约，得到 block 的全局最大，再由 lane 0 写回
    if (warp_id == 0) {
        // 每个 lane 持有一个候选
        T lane_val = lane_id < warp_per_block ? vals_shared[lane_id] : static_cast<T>(-INFINITY);
        int64_t lane_idx = lane_id < warp_per_block ? idxs_shared[lane_id] : -1;
        T final_val;
        int64_t final_idx;
        warp_argmax(lane_val, lane_idx, final_val, final_idx);
        if (lane_id == 0) {
            *max_val = final_val;
            *max_idx = final_idx;
        }
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void argmax(int64_t *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    // 特殊处理空张量的情况：max_val 是 std::byte*，需按类型写入
    if (numel == 0) {
        *max_idx = 0;
        switch (type) {
        case LLAISYS_DTYPE_F32:
            *reinterpret_cast<float *>(max_val) = 0.0f;
            break;
        case LLAISYS_DTYPE_F16:
            *reinterpret_cast<half *>(max_val) = __float2half(0.0f);
            break;
        case LLAISYS_DTYPE_BF16:
            *reinterpret_cast<__nv_bfloat16 *>(max_val) = __float2bfloat16(0.0f);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
        return;
    }

    const int block_size = 256;
    const int grid_size = 1;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<float, block_size><<<grid_size, block_size>>>(max_idx,
                                                                    reinterpret_cast<float *>(max_val),
                                                                    reinterpret_cast<const float *>(vals),
                                                                    numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<half, block_size><<<grid_size, block_size>>>(max_idx,
                                                                   reinterpret_cast<half *>(max_val),
                                                                   reinterpret_cast<const half *>(vals),
                                                                   numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<__nv_bfloat16, block_size><<<grid_size, block_size>>>(max_idx,
                                                                            reinterpret_cast<__nv_bfloat16 *>(max_val),
                                                                            reinterpret_cast<const __nv_bfloat16 *>(vals),
                                                                            numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
