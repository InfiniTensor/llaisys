#include "flash_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// ── 类型转换 helpers ──────────────────────────────────────────────────────────
template <typename T>
static inline float to_f32(T v) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::cast<float>(v);
    } else {
        return static_cast<float>(v);
    }
}

template <typename T>
static inline T from_f32(float v) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::cast<T>(v);
    } else {
        return static_cast<T>(v);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  DECODE 路径  (seq_len == 1)
//
//  每次生成一个 token 时 seq_len=1，对全部 KV 位置做标准 attention。
//  优化重点：
//  • 直接访问原始 KV 布局，零复制（消除旧实现的 K_h/V_h 转置拷贝）
//  • V 按行迭代（cache-friendly），消除旧实现的按列访问导致的 cache miss
//  • 缓冲区在 head 循环外分配一次，所有 head 复用
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
static void decode_attention_(T *out, const T *q, const T *k, const T *v,
                              size_t total_len,
                              size_t n_heads, size_t n_kv_heads, size_t head_dim,
                              float scale) {
    const size_t kv_stride = n_kv_heads * head_dim;
    const size_t heads_per_kv = n_heads / n_kv_heads;

    // head 循环外分配，所有 head 复用
    std::vector<float> scores(total_len);
    std::vector<float> out_f(head_dim);

    for (size_t h = 0; h < n_heads; h++) {
        const size_t kv_h = h / heads_per_kv;
        const T *q_row = q + h * head_dim; // q: [1, n_heads, head_dim]

        // ── QKᵀ ───────────────────────────────────────────────────────────
        for (size_t j = 0; j < total_len; j++) {
            const T *k_row = k + j * kv_stride + kv_h * head_dim;
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; d++) {
                dot += to_f32(q_row[d]) * to_f32(k_row[d]);
            }
            scores[j] = dot * scale;
        }

        // ── Softmax ───────────────────────────────────────────────────────
        float max_s = scores[0];
        for (size_t j = 1; j < total_len; j++) {
            if (scores[j] > max_s) {
                max_s = scores[j];
            }
        }
        float sum_e = 0.0f;
        for (size_t j = 0; j < total_len; j++) {
            scores[j] = std::exp(scores[j] - max_s);
            sum_e += scores[j];
        }
        const float inv_sum = 1.0f / sum_e;
        for (size_t j = 0; j < total_len; j++) {
            scores[j] *= inv_sum;
        }

        // ── O = attn · V  (按行迭代 V，顺序内存访问，L1 友好) ─────────────
        std::fill(out_f.begin(), out_f.end(), 0.0f);
        for (size_t j = 0; j < total_len; j++) {
            const float a_j = scores[j];
            const T *v_row = v + j * kv_stride + kv_h * head_dim;
            for (size_t d = 0; d < head_dim; d++) {
                out_f[d] += a_j * to_f32(v_row[d]);
            }
        }

        // ── 写回 ──────────────────────────────────────────────────────────
        T *out_row = out + h * head_dim;
        for (size_t d = 0; d < head_dim; d++) {
            out_row[d] = from_f32<T>(out_f[d]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  PREFILL 路径  (seq_len > 1)
//
//  与原版 self_attention_cpu 算法完全一致（标准两遍 softmax），
//  仅做内存访问优化：V 按行迭代（cache-friendly），attn_scores 复用缓冲区。
//  • Causal mask：第 i 个 query 可见位置 ≤ total_len - seq_len + i
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
static void prefill_attention_(T *out, const T *q, const T *k, const T *v,
                               size_t seq_len, size_t total_len,
                               size_t n_heads, size_t n_kv_heads, size_t head_dim,
                               float scale) {
    const size_t q_stride = n_heads * head_dim;
    const size_t kv_stride = n_kv_heads * head_dim;
    const size_t heads_per_kv = n_heads / n_kv_heads;

    // 在 head 循环外分配，所有 head 复用
    std::vector<float> attn_scores(seq_len * total_len);
    std::vector<float> out_f(head_dim);

    for (size_t h = 0; h < n_heads; h++) {
        const size_t kv_h = h / heads_per_kv;

        // ── S = Q · Kᵀ * scale，应用 causal mask ─────────────────────────
        for (size_t i = 0; i < seq_len; i++) {
            const size_t query_pos = total_len - seq_len + i; // 该 query 的绝对位置
            const T *q_row = q + i * q_stride + h * head_dim;

            for (size_t j = 0; j < total_len; j++) {
                if (j > query_pos) {
                    attn_scores[i * total_len + j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const T *k_row = k + j * kv_stride + kv_h * head_dim;
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    dot += to_f32(q_row[d]) * to_f32(k_row[d]);
                }
                attn_scores[i * total_len + j] = dot * scale;
            }
        }

        // ── 逐行 Softmax ──────────────────────────────────────────────────
        for (size_t i = 0; i < seq_len; i++) {
            float *row = attn_scores.data() + i * total_len;

            // 找最大值（数值稳定）
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < total_len; j++) {
                if (row[j] > max_score) {
                    max_score = row[j];
                }
            }

            // exp & sum（-inf → 0）
            float sum_exp = 0.0f;
            for (size_t j = 0; j < total_len; j++) {
                if (std::isinf(row[j]) && row[j] < 0.0f) {
                    row[j] = 0.0f;
                } else {
                    row[j] = std::exp(row[j] - max_score);
                    sum_exp += row[j];
                }
            }

            // 归一化
            const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            for (size_t j = 0; j < total_len; j++) {
                row[j] *= inv_sum;
            }
        }

        // ── O = attn · V  (按行迭代 V，L1 友好) ──────────────────────────
        for (size_t i = 0; i < seq_len; i++) {
            const float *a_row = attn_scores.data() + i * total_len;
            std::fill(out_f.begin(), out_f.end(), 0.0f);

            for (size_t j = 0; j < total_len; j++) {
                const float a_j = a_row[j];
                if (a_j == 0.0f) {
                    continue;
                }
                const T *v_row = v + j * kv_stride + kv_h * head_dim;
                for (size_t d = 0; d < head_dim; d++) {
                    out_f[d] += a_j * to_f32(v_row[d]);
                }
            }

            // 写回
            T *out_row = out + i * q_stride + h * head_dim;
            for (size_t d = 0; d < head_dim; d++) {
                out_row[d] = from_f32<T>(out_f[d]);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void flash_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                     llaisysDataType_t type,
                     size_t seq_len, size_t total_len,
                     size_t n_heads, size_t n_kv_heads, size_t head_dim, float scale) {

// 根据 seq_len 分路径：decode (seq_len==1) 与 prefill (seq_len>1)
#define FA_DISPATCH(CPP_TYPE)                                                             \
    do {                                                                                  \
        auto *out_ = reinterpret_cast<CPP_TYPE *>(out);                                   \
        const auto *q_ = reinterpret_cast<const CPP_TYPE *>(q);                           \
        const auto *k_ = reinterpret_cast<const CPP_TYPE *>(k);                           \
        const auto *v_ = reinterpret_cast<const CPP_TYPE *>(v);                           \
        if (seq_len == 1)                                                                 \
            decode_attention_(out_, q_, k_, v_,                                           \
                              total_len, n_heads, n_kv_heads, head_dim, scale);           \
        else                                                                              \
            prefill_attention_(out_, q_, k_, v_,                                          \
                               seq_len, total_len, n_heads, n_kv_heads, head_dim, scale); \
    } while (0)

    switch (type) {
    case LLAISYS_DTYPE_F32:
        FA_DISPATCH(float);
        return;
    case LLAISYS_DTYPE_BF16:
        FA_DISPATCH(llaisys::bf16_t);
        return;
    case LLAISYS_DTYPE_F16:
        FA_DISPATCH(llaisys::fp16_t);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
#undef FA_DISPATCH
}
} // namespace llaisys::ops::cpu
