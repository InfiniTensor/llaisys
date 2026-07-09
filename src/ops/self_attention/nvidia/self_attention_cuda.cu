#include "self_attention_cuda.cuh"

#include "../../../device/nvidia/nvidia_resource.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <sstream>
#include <stdexcept>

#define CUBLAS_CHECK(call)                                   \
    do {                                                     \
        cublasStatus_t _st = (call);                         \
        if (_st != CUBLAS_STATUS_SUCCESS) {                  \
            std::ostringstream _oss;                         \
            _oss << "cuBLAS error " << static_cast<int>(_st) \
                 << " at " << __FILE__ << ":" << __LINE__;   \
            throw std::runtime_error(_oss.str());            \
        }                                                    \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Warp reduce sum（用于 softmax）
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warpReduceSum_attn(float val) {
    constexpr unsigned FULL_MASK = 0xffffffff;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMax_attn(float val) {
    constexpr unsigned FULL_MASK = 0xffffffff;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

// ─────────────────────────────────────────────────────────────────────────────
// 因果 Softmax kernel
// gridDim.x = seq_len * n_heads（每行一个 block）
// 每行长度 = total_len
// ─────────────────────────────────────────────────────────────────────────────
__global__ void causal_softmax_kernel(float *__restrict__ attn_scores,
                                      size_t seq_len,
                                      size_t total_len,
                                      size_t n_heads) {
    extern __shared__ float smem[];

    size_t row = blockIdx.x; // row = query_pos * n_heads + head_id
    size_t q_idx = row / n_heads;
    // 当前 query 在序列中的绝对位置（total_len - seq_len + q_idx）
    size_t query_pos = total_len - seq_len + q_idx;

    float *row_ptr = attn_scores + row * total_len;

    int num_warps = (blockDim.x + 31) / 32;

    // ── 阶段1：找最大值（因果掩码 j > query_pos 置 -inf）────────────────
    float local_max = -3.402823466e+38f;
    for (size_t j = threadIdx.x; j < total_len; j += blockDim.x) {
        float v = (j <= query_pos) ? row_ptr[j] : -3.402823466e+38f;
        local_max = fmaxf(local_max, v);
    }
    local_max = warpReduceMax_attn(local_max);
    int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        smem[warp_id] = local_max;
    }
    __syncthreads();
    float block_max = (threadIdx.x < static_cast<unsigned>(num_warps)) ? smem[threadIdx.x] : -3.402823466e+38f;
    if (threadIdx.x < 32) {
        block_max = warpReduceMax_attn(block_max);
    }
    if (threadIdx.x == 0) {
        smem[0] = block_max;
    }
    __syncthreads();
    float max_val = smem[0];

    // ── 阶段2：计算 exp(x - max) 并求和 ──────────────────────────────
    float local_sum = 0.0f;
    for (size_t j = threadIdx.x; j < total_len; j += blockDim.x) {
        float v;
        if (j <= query_pos) {
            v = __expf(row_ptr[j] - max_val);
            row_ptr[j] = v;
        } else {
            row_ptr[j] = 0.0f;
            v = 0.0f;
        }
        local_sum += v;
    }
    local_sum = warpReduceSum_attn(local_sum);
    if (lane_id == 0) {
        smem[warp_id] = local_sum;
    }
    __syncthreads();
    float block_sum = (threadIdx.x < static_cast<unsigned>(num_warps)) ? smem[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        block_sum = warpReduceSum_attn(block_sum);
    }
    if (threadIdx.x == 0) {
        smem[0] = block_sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / (smem[0] + 1e-12f);

    // ── 阶段3：归一化 ────────────────────────────────────────────────
    for (size_t j = threadIdx.x; j < total_len; j += blockDim.x) {
        row_ptr[j] *= inv_sum;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 将 T 精度的 Q/K/V 提升到 float（用于中间计算），或保持 float 不变
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
__global__ void cast_to_f32_kernel(float *__restrict__ dst,
                                   const T *__restrict__ src,
                                   size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if constexpr (std::is_same_v<T, __half>) {
        dst[idx] = __half2float(src[idx]);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dst[idx] = __bfloat162float(src[idx]);
    } else {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

template <typename T>
__global__ void cast_from_f32_kernel(T *__restrict__ dst,
                                     const float *__restrict__ src,
                                     size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if constexpr (std::is_same_v<T, __half>) {
        dst[idx] = __float2half(src[idx]);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dst[idx] = __float2bfloat16(src[idx]);
    } else {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GQA/MHA self-attention 核心逻辑（在 float 精度下计算）
//
// Q[seq_len, n_heads, head_dim]  → 每个 Q head 对应一个 KV head
// K[total_len, n_kv_heads, head_dim]
// V[total_len, n_kv_heads, head_dim]
// O[seq_len, n_heads, head_dim]
//
// 对每个 head h：
//   kv_head = h / (n_heads / n_kv_heads)
//   Qh = Q[:, h, :]         [seq_len, head_dim]
//   Kh = K[:, kv_head, :]   [total_len, head_dim]
//   Vh = V[:, kv_head, :]   [total_len, head_dim]
//   Sh = Qh * Kh^T * scale  [seq_len, total_len]
//   Oh = softmax_causal(Sh) * Vh  [seq_len, head_dim]
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
static void self_attention_impl(float *out_f32,
                                const float *q_f32,
                                const float *k_f32,
                                const float *v_f32,
                                size_t seq_len, size_t total_len,
                                size_t n_heads, size_t n_kv_heads,
                                size_t head_dim, float scale) {
    cublasHandle_t handle = llaisys::device::nvidia::getCublasHandle();
    size_t heads_per_kv = n_heads / n_kv_heads;

    // 临时显存：注意力分数矩阵 [seq_len * total_len]（每 head 复用）
    float *d_scores = nullptr;
    cudaMalloc(&d_scores, seq_len * total_len * sizeof(float));

    for (size_t h = 0; h < n_heads; h++) {
        size_t kv_head = h / heads_per_kv;

        // Qh：步长 = n_heads * head_dim，偏移 = h * head_dim
        const float *Qh = q_f32 + h * head_dim;
        // Kh：步长 = n_kv_heads * head_dim，偏移 = kv_head * head_dim
        const float *Kh = k_f32 + kv_head * head_dim;
        // Vh：同 Kh 布局
        const float *Vh = v_f32 + kv_head * head_dim;
        // Oh：步长 = n_heads * head_dim，偏移 = h * head_dim
        float *Oh = out_f32 + h * head_dim;

        // ── S = Q * K^T * scale ────────────────────────────────────────────
        // Qh: [seq_len, head_dim] 行主序，stride = n_heads * head_dim
        // Kh: [total_len, head_dim] 行主序，stride = n_kv_heads * head_dim
        // S:  [seq_len, total_len] 行主序
        //
        // Row-major → cuBLAS 列主序变换：
        //   Kh (row-major) = Kh^T (列主序)，列步长 = n_kv_heads * head_dim
        //   Qh (row-major) = Qh^T (列主序)，列步长 = n_heads * head_dim
        //
        //   S^T[total_len x seq_len] = op(Kh^T) * op(Qh^T)
        //                            = CUBLAS_OP_T(Kh^T) * CUBLAS_OP_N(Qh^T)
        //                            = Kh * Qh^T  ✓
        //
        //   lda 约束：CUBLAS_OP_T 时 lda >= K = head_dim，n_kv_heads*head_dim >= head_dim ✓
        //   ldb 约束：CUBLAS_OP_N 时 ldb >= K = head_dim，n_heads*head_dim >= head_dim ✓
        //   ldc 约束：ldc >= M = total_len ✓（d_scores 连续分配）
        //
        int M = static_cast<int>(total_len);
        int N = static_cast<int>(seq_len);
        int K = static_cast<int>(head_dim);
        int lda = static_cast<int>(n_kv_heads * head_dim); // Kh^T 列步长
        int ldb = static_cast<int>(n_heads * head_dim);    // Qh^T 列步长
        int ldc = static_cast<int>(total_len);

        const float beta0 = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N, // ← 修正：原为 OP_N, OP_T
                                 M, N, K,
                                 &scale,
                                 Kh, lda,
                                 Qh, ldb,
                                 &beta0,
                                 d_scores, ldc));

        // ── 因果 Softmax ───────────────────────────────────────────────────
        {
            int blk = static_cast<int>(total_len < 1024 ? total_len : 1024);
            blk = ((blk + 31) / 32) * 32;
            int num_warps = (blk + 31) / 32;
            size_t smem = static_cast<size_t>(num_warps) * sizeof(float);
            // 每行 = 一个 query position × head = (q_idx * n_heads + h) 对应行
            // 但我们对每个 head 单独调用，scores 是 [seq_len, total_len]
            causal_softmax_kernel<<<static_cast<unsigned>(seq_len), blk, smem>>>(
                d_scores, seq_len, total_len, 1 /*n_heads=1, 已 per-head*/);
        }

        // ── O = softmax(S) * V ─────────────────────────────────────────────
        // softmax(S): [seq_len, total_len] 行主序
        // Vh:         [total_len, head_dim] 行主序，stride = n_kv_heads * head_dim
        // Oh:         [seq_len, head_dim] 行主序，stride = n_heads * head_dim
        //
        // cuBLAS：Oh^T = Vh * softmax(S)^T
        //   cublasSgemm(OP_N, OP_N, head_dim, seq_len, total_len,
        //               1, Vh, lda_v, S, total_len, 0, Oh, ldo)
        {
            int M2 = static_cast<int>(head_dim);
            int N2 = static_cast<int>(seq_len);
            int K2 = static_cast<int>(total_len);
            int lda2 = static_cast<int>(n_kv_heads * head_dim); // Vh row stride
            int ldb2 = static_cast<int>(total_len);             // S  row stride
            int ldc2 = static_cast<int>(n_heads * head_dim);    // Oh row stride

            const float alpha1 = 1.0f;
            CUBLAS_CHECK(cublasSgemm(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     M2, N2, K2,
                                     &alpha1,
                                     Vh, lda2,
                                     d_scores, ldb2,
                                     &beta0,
                                     Oh, ldc2));
        }
    }

    cudaFree(d_scores);
}

namespace llaisys::ops::nvidia {

void self_attention(std::byte *attn_val,
                    const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type,
                    size_t seq_len, size_t total_len,
                    size_t n_heads, size_t n_kv_heads,
                    size_t head_dim, float scale) {
    size_t q_numel = seq_len * n_heads * head_dim;
    size_t k_numel = total_len * n_kv_heads * head_dim;
    size_t v_numel = k_numel;
    size_t o_numel = q_numel;

    if (type == LLAISYS_DTYPE_F32) {
        // f32 直接传入，无需拷贝
        self_attention_impl<float>(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            seq_len, total_len, n_heads, n_kv_heads, head_dim, scale);
        return;
    }

    // f16/bf16：提升到 f32 后在中间计算，结果转回
    float *q_f32, *k_f32, *v_f32, *o_f32;
    cudaMalloc(&q_f32, q_numel * sizeof(float));
    cudaMalloc(&k_f32, k_numel * sizeof(float));
    cudaMalloc(&v_f32, v_numel * sizeof(float));
    cudaMalloc(&o_f32, o_numel * sizeof(float));

    constexpr int BLOCK = 256;
    auto grid = [&](size_t n) { return static_cast<int>((n + BLOCK - 1) / BLOCK); };

    switch (type) {
    case LLAISYS_DTYPE_F16:
        cast_to_f32_kernel<__half><<<grid(q_numel), BLOCK>>>(
            q_f32, reinterpret_cast<const __half *>(q), q_numel);
        cast_to_f32_kernel<__half><<<grid(k_numel), BLOCK>>>(
            k_f32, reinterpret_cast<const __half *>(k), k_numel);
        cast_to_f32_kernel<__half><<<grid(v_numel), BLOCK>>>(
            v_f32, reinterpret_cast<const __half *>(v), v_numel);
        break;
    case LLAISYS_DTYPE_BF16:
        cast_to_f32_kernel<__nv_bfloat16><<<grid(q_numel), BLOCK>>>(
            q_f32, reinterpret_cast<const __nv_bfloat16 *>(q), q_numel);
        cast_to_f32_kernel<__nv_bfloat16><<<grid(k_numel), BLOCK>>>(
            k_f32, reinterpret_cast<const __nv_bfloat16 *>(k), k_numel);
        cast_to_f32_kernel<__nv_bfloat16><<<grid(v_numel), BLOCK>>>(
            v_f32, reinterpret_cast<const __nv_bfloat16 *>(v), v_numel);
        break;
    default:
        cudaFree(q_f32);
        cudaFree(k_f32);
        cudaFree(v_f32);
        cudaFree(o_f32);
        throw std::runtime_error("self_attention CUDA: unsupported data type");
    }

    self_attention_impl<float>(
        o_f32, q_f32, k_f32, v_f32,
        seq_len, total_len, n_heads, n_kv_heads, head_dim, scale);

    switch (type) {
    case LLAISYS_DTYPE_F16:
        cast_from_f32_kernel<__half><<<grid(o_numel), BLOCK>>>(
            reinterpret_cast<__half *>(attn_val), o_f32, o_numel);
        break;
    case LLAISYS_DTYPE_BF16:
        cast_from_f32_kernel<__nv_bfloat16><<<grid(o_numel), BLOCK>>>(
            reinterpret_cast<__nv_bfloat16 *>(attn_val), o_f32, o_numel);
        break;
    default:
        break;
    }

    cudaFree(q_f32);
    cudaFree(k_f32);
    cudaFree(v_f32);
    cudaFree(o_f32);
}

} // namespace llaisys::ops::nvidia
