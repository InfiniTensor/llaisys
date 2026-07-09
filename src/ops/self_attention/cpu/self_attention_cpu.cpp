#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>

template <typename T>
void self_attention_(std::byte *attn_val_raw, const std::byte *q_raw, const std::byte *k_raw, const std::byte *v_raw, const size_t seqlen, const size_t nhead, const size_t nkvhead, const size_t d, const size_t dv, const size_t token_len, float scale) {
    T *out = reinterpret_cast<T*>(attn_val_raw);
    const T *q = reinterpret_cast<const T*>(q_raw);
    const T *k = reinterpret_cast<const T*>(k_raw);
    const T *v = reinterpret_cast<const T*>(v_raw);

    const size_t group_size = nhead / nkvhead;
    const size_t total_len = token_len;

    // 1. 遍历序列中的每一个 Query Token
    for (size_t i = 0; i < seqlen; ++i) {
        
        // 测试用例使用 tril(diagonal=S-L)，其中 L=seqlen, S=token_len
        // 允许的 key 位置满足 t <= i + (S - L)
        int64_t mask_limit = static_cast<int64_t>(i)
                           + static_cast<int64_t>(total_len)
                           - static_cast<int64_t>(seqlen);
        if (mask_limit >= static_cast<int64_t>(total_len)) {
            mask_limit = static_cast<int64_t>(total_len) - 1;
        }

        // 2. 遍历每一个 Attention Head
        for (size_t h = 0; h < nhead; ++h) {
            
            size_t kv_h = h / group_size;

            // 临时存储 Scores
            std::vector<float> scores(total_len, 0.0f);
            float max_score = -std::numeric_limits<float>::infinity();

            // --- Step 1: 计算 Q * K^T ---
            for (size_t t = 0; t < total_len; ++t) {
                // 因果掩码：只能看以前的 token
                // 如果 t > mask_limit，说明 Key 的位置在 Query 之后，屏蔽掉
                if (static_cast<int64_t>(t) > mask_limit) {
                    scores[t] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                // 点积计算
                float dot = 0.0f;
                const T* q_vec = q + (i * nhead * d) + (h * d);
                const T* k_vec = k + (t * nkvhead * d) + (kv_h * d);

                for (size_t m = 0; m < d; ++m) {
                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q_vec[m]);
                        k_val = llaisys::utils::cast<float>(k_vec[m]);
                    } else {
                        q_val = static_cast<float>(q_vec[m]);
                        k_val = static_cast<float>(k_vec[m]);
                    }
                    dot += q_val * k_val;
                }

                scores[t] = dot * scale;
                if (scores[t] > max_score) {
                    max_score = scores[t];
                }
            }

            // --- Step 2: Softmax ---
            float exp_sum = 0.0f;
            for (size_t t = 0; t < total_len; ++t) {
                if (static_cast<int64_t>(t) > mask_limit) {
                    scores[t] = 0.0f;
                } else {
                    float exp_val = std::exp(scores[t] - max_score);
                    scores[t] = exp_val;
                    exp_sum += exp_val;
                }
            }
            float inv_exp_sum = 1.0f / (exp_sum + 1e-9f);

            // --- Step 3: 加权求和 (prob * V) ---
            std::vector<float> acc(dv, 0.0f);
            for (size_t t = 0; t < total_len; ++t) {
                if (scores[t] == 0.0f) continue;
                
                float prob = scores[t] * inv_exp_sum;
                const T* v_vec = v + (t * nkvhead * dv) + (kv_h * dv);

                for (size_t m = 0; m < dv; ++m) {
                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v_vec[m]);
                    } else {
                        v_val = static_cast<float>(v_vec[m]);
                    }
                    acc[m] += prob * v_val;
                }
            }

            // --- Step 4: 写入输出 ---
            T* out_vec = out + (i * nhead * dv) + (h * dv);
            for (size_t m = 0; m < dv; ++m) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out_vec[m] = llaisys::utils::cast<T>(acc[m]);
                } else {
                    out_vec[m] = static_cast<T>(acc[m]);
                }
            }
        }
    }
}

#define DISPATCH_SELF_ATTENTION(dtype, ctype) case dtype: self_attention_<ctype>(attn_val, q, k, v, seqlen, nhead, nkvhead, d, dv, token_len, scale); break;

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, const size_t seqlen, const size_t nhead, const size_t nkvhead, const size_t d, const size_t dv, const size_t token_len, float scale){
    switch (type) {
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_F32, float)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_BF16, llaisys::bf16_t)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_F16, llaisys::fp16_t)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_I32, int32_t)
        DISPATCH_SELF_ATTENTION(LLAISYS_DTYPE_F64, double)
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
