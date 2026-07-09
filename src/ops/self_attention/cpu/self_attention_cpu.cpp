#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_kernel(const T *q, const T *k, const T *v, T *out,
                           size_t seqlen, size_t total_len,
                           size_t nhead, size_t nkvhead,
                           size_t head_dim, size_t head_dim_v,
                           float scale) {

    size_t group_size = nhead / nkvhead;

    for (size_t i = 0; i < seqlen; i++) {
        for (size_t h = 0; h < nhead; h++) {

            size_t q_offset = i * (nhead * head_dim) + h * head_dim;
            const T *q_vec = q + q_offset;

            size_t kv_h = h / group_size;

            std::vector<float> scores(total_len);
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t t = 0; t < total_len; t++) {

                size_t global_q_idx = (total_len - seqlen) + i;

                if (t > global_q_idx) {
                    scores[t] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                // 定位 K 向量 [t, kv_h, head_dim]
                size_t k_offset = t * (nkvhead * head_dim) + kv_h * head_dim;
                const T *k_vec = k + k_offset;

                // 点积
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    float q_val = llaisys::utils::cast<float>(q_vec[d]);
                    float k_val = llaisys::utils::cast<float>(k_vec[d]);
                    dot += q_val * k_val;
                }

                // 缩放
                scores[t] = dot * scale;

                // 记录最大值用于 Softmax 数值稳定
                if (scores[t] > max_score) {
                    max_score = scores[t];
                }
            }

            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_len; t++) {
                if (scores[t] == -std::numeric_limits<float>::infinity()) {
                    scores[t] = 0.0f; // exp(-inf) = 0
                } else {
                    // 减去最大值防止溢出
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
            }

            // 归一化
            for (size_t t = 0; t < total_len; t++) {
                scores[t] /= sum_exp;
            }

            size_t out_offset = i * (nhead * head_dim_v) + h * head_dim_v;
            T *out_vec = out + out_offset;

            for (size_t dv = 0; dv < head_dim_v; dv++) {
                float acc = 0.0f;
                for (size_t t = 0; t < total_len; t++) {

                    if (scores[t] == 0.0f) {
                        continue;
                    }

                    size_t v_offset = t * (nkvhead * head_dim_v) + kv_h * head_dim_v;
                    float v_val = llaisys::utils::cast<float>(v[v_offset + dv]);

                    acc += scores[t] * v_val;
                }
                out_vec[dv] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}

namespace llaisys::ops::cpu {

void self_attention(const std::byte *q_data, const std::byte *k_data, const std::byte *v_data,
                    std::byte *attn_val_data, llaisysDataType_t dtype,
                    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                    size_t head_dim, size_t head_dim_v, float scale) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<float>(
            reinterpret_cast<const float *>(q_data),
            reinterpret_cast<const float *>(k_data),
            reinterpret_cast<const float *>(v_data),
            reinterpret_cast<float *>(attn_val_data),
            seqlen, total_len, nhead, nkvhead, head_dim, head_dim_v, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<llaisys::fp16_t>(
            reinterpret_cast<const llaisys::fp16_t *>(q_data),
            reinterpret_cast<const llaisys::fp16_t *>(k_data),
            reinterpret_cast<const llaisys::fp16_t *>(v_data),
            reinterpret_cast<llaisys::fp16_t *>(attn_val_data),
            seqlen, total_len, nhead, nkvhead, head_dim, head_dim_v, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<llaisys::bf16_t>(
            reinterpret_cast<const llaisys::bf16_t *>(q_data),
            reinterpret_cast<const llaisys::bf16_t *>(k_data),
            reinterpret_cast<const llaisys::bf16_t *>(v_data),
            reinterpret_cast<llaisys::bf16_t *>(attn_val_data),
            seqlen, total_len, nhead, nkvhead, head_dim, head_dim_v, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu