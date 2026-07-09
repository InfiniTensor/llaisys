// Flash Attention 2 CUDA 实现（修正版）
//
// 修正记录：
//   1. FA_BC_CUDA 32→16：smem = (2×16+2×16)×128×4 + 2×16×4 = 32896B ≈ 32KB
//      原 FA_BC=32 时 smem=49280B 超过 48KB 默认上限，kernel 静默失败不写出，
//      输出 tensor 保持未初始化状态 → 纯随机乱码
//   2. 使用 __int_as_float(0xff800000) 作为真正的 IEEE-754 -inf（替代 -1e30f）
//      原 -1e30f：exp(-1e30-(-1e30))=exp(0)=1，masked 位置被错赋权重 1
//   3. 全 mask tile：m_new 仍为 -inf 时显式跳过，防止 -inf-(-inf)=NaN 传播
//   4. m_old==-inf 时 alpha 显式置 0，防止 exp(-inf) 精度边界问题
//   5. V 累积改为 j 外层 d 内层，V_s 顺序内存访问，减少 bank conflict

#include "flash_attention_cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

// ─── 精度辅助函数 ─────────────────────────────────────────────────────────────
__device__ __forceinline__ float to_f32_fa(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f32_fa(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f32_fa(float v) { return v; }

__device__ __forceinline__ __half from_f32_fa_h(float v) { return __float2half(v); }
__device__ __forceinline__ __nv_bfloat16 from_f32_fa_b(float v) { return __float2bfloat16(v); }

// ─── tile 参数 ────────────────────────────────────────────────────────────────
// FA_BC_CUDA=16 确保对 head_dim≤128 时 smem ≈ 32KB < 48KB（所有 CUDA GPU 安全）
// 原 FA_BC=32：smem = (16+64+16)×128×4 + 128 = 49280B > 48KB，kernel 静默失败
static constexpr int FA_BR_CUDA = 16; // Q  tile 行数（= blockDim.x）
static constexpr int FA_BC_CUDA = 16; // KV tile 列数（从 32 降为 16）

// ─── Flash Attention 2 核心 kernel ───────────────────────────────────────────
// grid  = (n_heads, ceil(seq_len / FA_BR))
// block = (FA_BR, 1)   — 每线程负责 Q-tile 中一行 query
template <typename T>
__global__ void flash_attn_kernel(
    T *__restrict__ out,     // [seq_len,  n_heads,    head_dim]
    const T *__restrict__ Q, // [seq_len,  n_heads,    head_dim]
    const T *__restrict__ K, // [total_len, n_kv_heads, head_dim]
    const T *__restrict__ V, // [total_len, n_kv_heads, head_dim]
    int seq_len, int total_len,
    int n_heads, int n_kv_heads, int head_dim, float scale,
    int heads_per_kv) {
    // IEEE-754 负无穷，通过位模式构造，避免 __builtin_inff() 在某些 CUDA 版本的行为差异
    const float NEG_INF = __int_as_float(0xff800000u);

    const int h = blockIdx.x;
    const int kv_h = h / heads_per_kv;
    const int q_start = blockIdx.y * FA_BR_CUDA;
    if (q_start >= seq_len) {
        return;
    }

    const int q_end = min(q_start + FA_BR_CUDA, seq_len);
    const int q_len = q_end - q_start;
    const int tx = threadIdx.x; // 0 .. FA_BR_CUDA-1

    // ── Shared memory 布局 ───────────────────────────────────────────────────
    // Q_s [FA_BR, head_dim]  K_s [FA_BC, head_dim]
    // V_s [FA_BC, head_dim]  O_s [FA_BR, head_dim]
    // m_s [FA_BR]            l_s [FA_BR]
    extern __shared__ float smem[];
    float *Q_s = smem;
    float *K_s = Q_s + FA_BR_CUDA * head_dim;
    float *V_s = K_s + FA_BC_CUDA * head_dim;
    float *O_s = V_s + FA_BC_CUDA * head_dim;
    float *m_s = O_s + FA_BR_CUDA * head_dim;
    float *l_s = m_s + FA_BR_CUDA;

    // ── 初始化：载入 Q tile，重置 O/m/l ─────────────────────────────────────
    if (tx < q_len) {
        m_s[tx] = NEG_INF; // 真正的 -inf，确保第一有效 tile 时 alpha=0
        l_s[tx] = 0.0f;

        const int abs_q = q_start + tx;
        const T *q_ptr = Q + abs_q * n_heads * head_dim + h * head_dim;
        float *q_row = Q_s + tx * head_dim;
        float *o_row = O_s + tx * head_dim;
        for (int d = 0; d < head_dim; d++) {
            q_row[d] = to_f32_fa(q_ptr[d]);
            o_row[d] = 0.0f;
        }
    }
    __syncthreads();

    // ── 遍历 KV tiles ─────────────────────────────────────────────────────────
    for (int kv_start = 0; kv_start < total_len; kv_start += FA_BC_CUDA) {
        const int kv_end = min(kv_start + FA_BC_CUDA, total_len);
        const int kv_len = kv_end - kv_start;

        // 协作加载 K/V tile（FA_BR 个线程分摊 FA_BC 行）
        for (int row = tx; row < kv_len; row += FA_BR_CUDA) {
            const int abs_kv = kv_start + row;
            const T *k_ptr = K + abs_kv * n_kv_heads * head_dim + kv_h * head_dim;
            const T *v_ptr = V + abs_kv * n_kv_heads * head_dim + kv_h * head_dim;
            float *k_row = K_s + row * head_dim;
            float *v_row = V_s + row * head_dim;
            for (int d = 0; d < head_dim; d++) {
                k_row[d] = to_f32_fa(k_ptr[d]);
                v_row[d] = to_f32_fa(v_ptr[d]);
            }
        }
        __syncthreads();

        // ── 每线程独立处理其负责的 query 行 ─────────────────────────────────
        if (tx < q_len) {
            const int abs_q = q_start + tx;
            const int causal_lim = total_len - seq_len + abs_q; // causal mask 上界（含）

            // ① 计算 score tile 并找非 mask 位置最大值
            float s_local[FA_BC_CUDA]; // FA_BC_CUDA=16 → 64B，保持在寄存器中
            float local_max = NEG_INF;

            const float *q_row = Q_s + tx * head_dim;
            for (int j = 0; j < kv_len; j++) {
                const int abs_kv = kv_start + j;
                if (abs_kv > causal_lim) {
                    s_local[j] = NEG_INF; // 精确 -inf，exp(-inf)=0
                    continue;
                }
                const float *k_row = K_s + j * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_row[d] * k_row[d];
                }
                s_local[j] = dot * scale;
                if (s_local[j] > local_max) {
                    local_max = s_local[j];
                }
            }

            // ② Online softmax 更新
            const float m_old = m_s[tx];
            const float m_new = (local_max > m_old) ? local_max : m_old;

            // 若 m_new 仍为 -inf：本 tile 全被 causal mask，且历史无有效分数
            // → O/l/m 保持不变，直接跳过，避免 -inf-(-inf)=NaN
            if (!(__isinff(m_new) && m_new < 0.0f)) {

                // alpha：历史累积 O 的缩放系数
                // m_old==-inf 表示 O 尚未写入任何值，alpha 直接为 0
                const float alpha = __isinff(m_old) ? 0.0f : __expf(m_old - m_new);

                // ③ 计算 exp(s - m_new)，mask 位置显式置 0
                float l_tile = 0.0f;
                for (int j = 0; j < kv_len; j++) {
                    const float e = (__isinff(s_local[j]) && s_local[j] < 0.0f)
                                      ? 0.0f
                                      : __expf(s_local[j] - m_new);
                    s_local[j] = e;
                    l_tile += e;
                }

                // ④ 更新 O：缩放旧值 + 累积本 tile 的 V 贡献
                float *o_row = O_s + tx * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    o_row[d] *= alpha;
                }
                // j 外 d 内：V_s[j*hd+d] 顺序读取，减少 bank conflict
                for (int j = 0; j < kv_len; j++) {
                    const float sj = s_local[j];
                    if (sj == 0.0f) {
                        continue;
                    }
                    const float *v_row = V_s + j * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] += sj * v_row[d];
                    }
                }

                l_s[tx] = alpha * l_s[tx] + l_tile;
                m_s[tx] = m_new;
            }
            // else: 全 mask tile → O/l/m 不变
        }
        __syncthreads();
    }

    // ── 最终归一化并写回全局内存 ─────────────────────────────────────────────
    if (tx < q_len) {
        const int abs_q = q_start + tx;
        const float inv_l = (l_s[tx] > 0.0f) ? (1.0f / l_s[tx]) : 0.0f;
        const float *o_row = O_s + tx * head_dim;
        T *o_ptr = out + abs_q * n_heads * head_dim + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            if constexpr (std::is_same_v<T, float>) {
                o_ptr[d] = o_row[d] * inv_l;
            } else if constexpr (std::is_same_v<T, __half>) {
                o_ptr[d] = from_f32_fa_h(o_row[d] * inv_l);
            } else { // __nv_bfloat16
                o_ptr[d] = from_f32_fa_b(o_row[d] * inv_l);
            }
        }
    }
}

// ─── kernel 启动封装 ──────────────────────────────────────────────────────────
template <typename T>
static void launch_flash_attn(T *out, const T *q, const T *k, const T *v,
                              int seq_len, int total_len,
                              int n_heads, int n_kv_heads, int head_dim, float scale) {
    const int heads_per_kv = n_heads / n_kv_heads;
    const int num_q_tiles = (seq_len + FA_BR_CUDA - 1) / FA_BR_CUDA;

    dim3 grid(n_heads, num_q_tiles);
    dim3 block(FA_BR_CUDA, 1);

    // smem = (2×BR + 2×BC) × head_dim × 4 + 2×BR × 4
    // BR=BC=16, head_dim=128: (32+32)×512 + 128 = 32896B ≈ 32KB < 48KB ✓
    const size_t smem = static_cast<size_t>(2 * FA_BR_CUDA + 2 * FA_BC_CUDA) * head_dim * sizeof(float)
                      + static_cast<size_t>(2 * FA_BR_CUDA) * sizeof(float);

    flash_attn_kernel<T><<<grid, block, smem>>>(
        out, q, k, v, seq_len, total_len,
        n_heads, n_kv_heads, head_dim, scale, heads_per_kv);
}

namespace llaisys::ops::nvidia {

void flash_attention(std::byte *attn_val,
                     const std::byte *q, const std::byte *k, const std::byte *v,
                     llaisysDataType_t type,
                     size_t seq_len, size_t total_len,
                     size_t n_heads, size_t n_kv_heads,
                     size_t head_dim, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_flash_attn(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            static_cast<int>(seq_len), static_cast<int>(total_len),
            static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
            static_cast<int>(head_dim), scale);
        break;
    case LLAISYS_DTYPE_F16:
        launch_flash_attn(
            reinterpret_cast<__half *>(attn_val),
            reinterpret_cast<const __half *>(q),
            reinterpret_cast<const __half *>(k),
            reinterpret_cast<const __half *>(v),
            static_cast<int>(seq_len), static_cast<int>(total_len),
            static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
            static_cast<int>(head_dim), scale);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_flash_attn(
            reinterpret_cast<__nv_bfloat16 *>(attn_val),
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            static_cast<int>(seq_len), static_cast<int>(total_len),
            static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
            static_cast<int>(head_dim), scale);
        break;
    default:
        throw std::runtime_error("flash_attention CUDA: unsupported data type");
    }
}

} // namespace llaisys::ops::nvidia
