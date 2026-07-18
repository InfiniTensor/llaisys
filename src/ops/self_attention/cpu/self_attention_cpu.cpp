#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>
#include <type_traits>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     float scale, size_t seq_len, size_t total_len,
                     size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    using acc_t = std::conditional_t<
        std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>,
        float, T>;

    size_t n_rep = nhead / nkvhead;
    std::vector<acc_t> scores(total_len);

    for (size_t i = 0; i < seq_len; ++i) {
        size_t global_pos = total_len - seq_len + i;
        size_t valid_len = std::min(global_pos + 1, total_len);

        for (size_t h = 0; h < nhead; ++h) {
            size_t kv_h = h / n_rep;

            const T *q_ptr = q + (i * nhead * d) + (h * d);
            T *out_ptr = attn_val + (i * nhead * dv) + (h * dv);

            acc_t max_val = -std::numeric_limits<acc_t>::infinity();

            // 1. Calculate Scores (Only valid positions)
            for (size_t t = 0; t < valid_len; ++t) {
                const T *k_ptr = k + (t * nkvhead * d) + (kv_h * d);
                acc_t dot = 0;
                for (size_t j = 0; j < d; ++j) {
                    dot += llaisys::utils::cast<acc_t>(q_ptr[j]) *
                           llaisys::utils::cast<acc_t>(k_ptr[j]);
                }
                scores[t] = dot * scale;
                if (scores[t] > max_val) {
                    max_val = scores[t];
                }
            }

            // 2. Softmax (Only valid positions)
            acc_t sum_exp = 0;
            for (size_t t = 0; t < valid_len; ++t) {
                scores[t] = std::exp(scores[t] - max_val);
                sum_exp += scores[t];
            }

            // 3. Weighted Sum (Only valid positions)
            for (size_t j = 0; j < dv; ++j) {
                acc_t val = 0;
                for (size_t t = 0; t < valid_len; ++t) {
                    const T *v_ptr = v + (t * nkvhead * dv) + (kv_h * dv);
                    val += scores[t] * llaisys::utils::cast<acc_t>(v_ptr[j]);
                }
                out_ptr[j] = llaisys::utils::cast<T>(val / sum_exp);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t dtype,
                    size_t seq_len, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               scale, seq_len, total_len, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               scale, seq_len, total_len, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               scale, seq_len, total_len, nhead, nkvhead, d, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
