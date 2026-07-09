#include "rms_norm_cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Warp-level sum 规约（A100 warp size = 32）
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warpReduceSum(float val) {
    constexpr unsigned FULL_MASK = 0xffffffff;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// ─────────────────────────────────────────────────────────────────────────────
// RMS Norm kernel
// gridDim.x = num_rows；每个 block 使用最多 1024 个线程处理一行
// 共享内存用于 warp 间规约中间结果
// ─────────────────────────────────────────────────────────────────────────────
template <typename T, typename AccT = float>
__global__ void rms_norm_kernel(T *__restrict__ out,
                                const T *__restrict__ in,
                                const T *__restrict__ weight,
                                size_t row_dim,
                                float eps) {
    extern __shared__ float smem[]; // [num_warps] 个 float

    const size_t row = blockIdx.x;
    const T *in_row = in + row * row_dim;
    T *out_row = out + row * row_dim;

    // ── 阶段1：每线程并行累加 x^2 ──────────────────────────────────────────
    float local_sum = 0.0f;
    for (size_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
        float v;
        if constexpr (std::is_same_v<T, __half>) {
            v = __half2float(in_row[i]);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            v = __bfloat162float(in_row[i]);
        } else {
            v = static_cast<float>(in_row[i]);
        }
        local_sum += v * v;
    }

    // ── 阶段2：warp 内规约 ────────────────────────────────────────────────
    local_sum = warpReduceSum(local_sum);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = (blockDim.x + 31) / 32;

    if (lane_id == 0) {
        smem[warp_id] = local_sum;
    }
    __syncthreads();

    // ── 阶段3：跨 warp 规约（由第一个 warp 完成）────────────────────────
    float block_sum = 0.0f;
    if (threadIdx.x < static_cast<unsigned>(num_warps)) {
        block_sum = smem[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        block_sum = warpReduceSum(block_sum);
    }
    if (threadIdx.x == 0) {
        smem[0] = block_sum;
    }
    __syncthreads();

    // ── 阶段4：计算归一化因子 rms_inv ─────────────────────────────────────
    float rms_inv = rsqrtf(smem[0] / static_cast<float>(row_dim) + eps);

    // ── 阶段5：写出归一化结果 ─────────────────────────────────────────────
    for (size_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
        float x_val, w_val;
        if constexpr (std::is_same_v<T, __half>) {
            x_val = __half2float(in_row[i]);
            w_val = __half2float(weight[i]);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            x_val = __bfloat162float(in_row[i]);
            w_val = __bfloat162float(weight[i]);
        } else {
            x_val = static_cast<float>(in_row[i]);
            w_val = static_cast<float>(weight[i]);
        }
        float result = x_val * rms_inv * w_val;

        if constexpr (std::is_same_v<T, __half>) {
            out_row[i] = __float2half(result);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            out_row[i] = __float2bfloat16(result);
        } else {
            out_row[i] = static_cast<T>(result);
        }
    }
}

namespace llaisys::ops::nvidia {

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t num_rows, size_t row_dim, float eps) {
    if (num_rows == 0 || row_dim == 0) {
        return;
    }

    // block 大小：clamp 到 1024，按 32 对齐
    int block = static_cast<int>(row_dim < 1024 ? row_dim : 1024);
    block = ((block + 31) / 32) * 32; // round up to warp
    int num_warps = (block + 31) / 32;
    size_t smem_bytes = static_cast<size_t>(num_warps) * sizeof(float);

    dim3 grid(static_cast<unsigned>(num_rows));
    dim3 blk(static_cast<unsigned>(block));

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float><<<grid, blk, smem_bytes>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            row_dim, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<__half><<<grid, blk, smem_bytes>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight),
            row_dim, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<__nv_bfloat16><<<grid, blk, smem_bytes>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            row_dim, eps);
        break;
    default:
        throw std::runtime_error("rms_norm CUDA: unsupported data type");
    }
}

} // namespace llaisys::ops::nvidia
