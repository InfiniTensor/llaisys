#include "argmax_cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Warp-level (val, idx) max 规约
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void warpReduceMax(float &val, int &idx) {
    constexpr unsigned FULL_MASK = 0xffffffff;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(FULL_MASK, val, offset);
        int other_idx = __shfl_down_sync(FULL_MASK, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 每个 block 找到自己负责段的最大值/索引，然后写入共享内存
// 第二阶段由单 block 归约（block_size = 1 启动）完成全局 argmax
// 对于推理中 vocab 规模（~32k~128k），通常一次 launch 就够
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
__global__ void argmax_kernel(const T *__restrict__ vals,
                              size_t n,
                              float *__restrict__ blk_max_val,
                              int *__restrict__ blk_max_idx) {
    extern __shared__ char smem_raw[];
    float *smem_val = reinterpret_cast<float *>(smem_raw);
    int *smem_idx = reinterpret_cast<int *>(smem_val + (blockDim.x / 32));

    float local_val = -3.402823466e+38f; // -FLT_MAX
    int local_idx = 0;

    // 每线程在自己的条带范围内找局部最大
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float v;
        if constexpr (std::is_same_v<T, __half>) {
            v = __half2float(vals[i]);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            v = __bfloat162float(vals[i]);
        } else {
            v = static_cast<float>(vals[i]);
        }
        if (v > local_val) {
            local_val = v;
            local_idx = static_cast<int>(i);
        }
    }

    // Warp reduce
    warpReduceMax(local_val, local_idx);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = (blockDim.x + 31) / 32;

    if (lane_id == 0) {
        smem_val[warp_id] = local_val;
        smem_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // Block reduce（第一个 warp 完成跨 warp 归约）
    float bval = -3.402823466e+38f;
    int bidx = 0;
    if (threadIdx.x < static_cast<unsigned>(num_warps)) {
        bval = smem_val[threadIdx.x];
        bidx = smem_idx[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        warpReduceMax(bval, bidx);
    }
    if (threadIdx.x == 0) {
        blk_max_val[blockIdx.x] = bval;
        blk_max_idx[blockIdx.x] = bidx;
    }
}

// 最终全局归约（在 CPU 端完成，或用单 block kernel）
__global__ void argmax_final_kernel(const float *__restrict__ blk_val,
                                    const int *__restrict__ blk_idx,
                                    int num_blocks,
                                    int64_t *__restrict__ out_idx,
                                    float *__restrict__ out_val) {
    float best_val = -3.402823466e+38f;
    int best_idx = 0;
    for (int i = 0; i < num_blocks; i++) {
        if (blk_val[i] > best_val) {
            best_val = blk_val[i];
            best_idx = blk_idx[i];
        }
    }
    out_idx[0] = static_cast<int64_t>(best_idx);
    out_val[0] = best_val;
}

namespace llaisys::ops::nvidia {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel) {
    constexpr int BLOCK = 256;
    int num_blocks = static_cast<int>((numel + BLOCK - 1) / BLOCK);
    // 限制 block 数，避免显存申请过多
    if (num_blocks > 512) {
        num_blocks = 512;
    }

    int num_warps = (BLOCK + 31) / 32;
    size_t smem = static_cast<size_t>(num_warps) * (sizeof(float) + sizeof(int));

    // 临时显存存储每个 block 的局部结果
    float *d_blk_val = nullptr;
    int *d_blk_idx = nullptr;
    cudaMalloc(&d_blk_val, num_blocks * sizeof(float));
    cudaMalloc(&d_blk_idx, num_blocks * sizeof(int));

    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<float><<<num_blocks, BLOCK, smem>>>(
            reinterpret_cast<const float *>(vals), numel, d_blk_val, d_blk_idx);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<__half><<<num_blocks, BLOCK, smem>>>(
            reinterpret_cast<const __half *>(vals), numel, d_blk_val, d_blk_idx);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<__nv_bfloat16><<<num_blocks, BLOCK, smem>>>(
            reinterpret_cast<const __nv_bfloat16 *>(vals), numel, d_blk_val, d_blk_idx);
        break;
    default:
        cudaFree(d_blk_val);
        cudaFree(d_blk_idx);
        throw std::runtime_error("argmax CUDA: unsupported data type");
    }

    // 单线程最终归约（num_blocks 通常 ≤ 512，可在单线程完成）
    argmax_final_kernel<<<1, 1>>>(
        d_blk_val, d_blk_idx, num_blocks,
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<float *>(max_val));

    cudaFree(d_blk_val);
    cudaFree(d_blk_idx);
}

} // namespace llaisys::ops::nvidia
