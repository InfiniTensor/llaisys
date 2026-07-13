#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     float scale, size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t head_dim) {
    // Calculate the group size for GQA (Grouped Query Attention)
    // Each kv head is shared by (nhead / nkvhead) query heads
    const size_t group_size = nhead / nkvhead;
    
    // For each query position
    for (size_t qi = 0; qi < qlen; ++qi) {
        // For each attention head
        for (size_t h = 0; h < nhead; ++h) {
            // Find the corresponding kv head for this query head
            const size_t kv_h = h / group_size;
            
            // Compute attention scores for this query position and head
            // Q[qi, h, :] @ K[:, kv_h, :].T -> scores[kvlen]
            std::vector<double> scores(kvlen);
            
            // Calculate Q @ K^T * scale
            for (size_t ki = 0; ki < kvlen; ++ki) {
                double score = 0.0;
                for (size_t d = 0; d < head_dim; ++d) {
                    // q index: [qi, h, d] -> qi * nhead * head_dim + h * head_dim + d
                    // k index: [ki, kv_h, d] -> ki * nkvhead * head_dim + kv_h * head_dim + d
                    double q_val = llaisys::utils::cast<double>(q[qi * nhead * head_dim + h * head_dim + d]);
                    double k_val = llaisys::utils::cast<double>(k[ki * nkvhead * head_dim + kv_h * head_dim + d]);
                    score += q_val * k_val;
                }
                scores[ki] = score * static_cast<double>(scale);
            }
            
            // Apply causal mask: for position qi in query, we can only attend to positions 0..qi+(kvlen-qlen)
            // The causal mask is: temp_mask = torch.ones(L, S).tril(diagonal=S-L)
            // attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            // This means: for query position qi, we can attend to key positions 0 to (kvlen - qlen + qi)
            const size_t max_attend = kvlen - qlen + qi;
            for (size_t ki = max_attend + 1; ki < kvlen; ++ki) {
                scores[ki] = -std::numeric_limits<double>::infinity();
            }
            
            // Compute softmax
            // Find max for numerical stability
            double max_score = scores[0];
            for (size_t ki = 1; ki < kvlen; ++ki) {
                if (scores[ki] > max_score) {
                    max_score = scores[ki];
                }
            }
            
            // Compute exp and sum
            double sum_exp = 0.0;
            for (size_t ki = 0; ki < kvlen; ++ki) {
                scores[ki] = std::exp(scores[ki] - max_score);
                sum_exp += scores[ki];
            }
            
            // Normalize
            for (size_t ki = 0; ki < kvlen; ++ki) {
                scores[ki] /= sum_exp;
            }
            
            // Compute attention output: softmax(scores) @ V
            // attn_val[qi, h, :] = sum over ki of (scores[ki] * V[ki, kv_h, :])
            for (size_t d = 0; d < head_dim; ++d) {
                double out_val = 0.0;
                for (size_t ki = 0; ki < kvlen; ++ki) {
                    // v index: [ki, kv_h, d] -> ki * nkvhead * head_dim + kv_h * head_dim + d
                    double v_val = llaisys::utils::cast<double>(v[ki * nkvhead * head_dim + kv_h * head_dim + d]);
                    out_val += scores[ki] * v_val;
                }
                // attn_val index: [qi, h, d] -> qi * nhead * head_dim + h * head_dim + d
                attn_val[qi * nhead * head_dim + h * head_dim + d] = llaisys::utils::cast<T>(out_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t type,
                    size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t head_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               scale, qlen, kvlen, nhead, nkvhead, head_dim);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               scale, qlen, kvlen, nhead, nkvhead, head_dim);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               scale, qlen, kvlen, nhead, nkvhead, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
