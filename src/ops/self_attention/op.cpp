#include "op.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace llaisys::ops {
template <typename T>
void self_attention_cpu_(T *attn_val, const T *q, const T *k, const T *v,
                         float scale, size_t seqlen, size_t nhead, size_t nkvhead,
                         size_t d, size_t dv, size_t total_len) {
    // Compute: Y = causal_softmax(Q @ K^T * scale) @ V
    // q: [seqlen, nhead, d]
    // k: [total_len, nkvhead, d]
    // v: [total_len, nkvhead, dv]
    // attn_val: [seqlen, nhead, dv]
    
    // Group size for GQA (Grouped Query Attention)
    size_t group_size = nhead / nkvhead;
    
    for (size_t s = 0; s < seqlen; s++) {         // For each query position
        for (size_t h = 0; h < nhead; h++) {      // For each query head
            // Determine which KV head to use (for GQA)
            size_t kv_h = h / group_size;
            
            // Allocate temporary buffer for attention scores
            std::vector<float> attn_scores(total_len);
            
            // Step 1: Compute Q @ K^T for this query
            for (size_t t = 0; t < total_len; t++) {
                float score = 0.0f;
                
                for (size_t i = 0; i < d; i++) {
                    size_t q_idx = s * nhead * d + h * d + i;
                    size_t k_idx = t * nkvhead * d + kv_h * d + i;
                    
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        float q_val = llaisys::utils::cast<float>(q[q_idx]);
                        float k_val = llaisys::utils::cast<float>(k[k_idx]);
                        score += q_val * k_val;
                    } else {
                        score += static_cast<float>(q[q_idx]) * static_cast<float>(k[k_idx]);
                    }
                }
                
                attn_scores[t] = score * scale;
            }
            
            // Step 2: Apply causal mask and softmax
            // Causal mask: can only attend to positions <= current position + offset
            // offset = total_len - seqlen (for KV cache)
            size_t offset = total_len - seqlen;
            size_t current_pos = s + offset;
            
            // Find max for numerical stability
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t t = 0; t <= current_pos && t < total_len; t++) {
                max_score = std::max(max_score, attn_scores[t]);
            }
            
            // Compute exp and sum
            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_len; t++) {
                if (t <= current_pos) {
                    attn_scores[t] = std::exp(attn_scores[t] - max_score);
                    sum_exp += attn_scores[t];
                } else {
                    attn_scores[t] = 0.0f;  // Masked positions
                }
            }
            
            // Normalize
            for (size_t t = 0; t < total_len; t++) {
                attn_scores[t] /= sum_exp;
            }
            
            // Step 3: Compute weighted sum: attn_scores @ V
            for (size_t j = 0; j < dv; j++) {
                float result = 0.0f;
                
                for (size_t t = 0; t < total_len; t++) {
                    size_t v_idx = t * nkvhead * dv + kv_h * dv + j;
                    
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        float v_val = llaisys::utils::cast<float>(v[v_idx]);
                        result += attn_scores[t] * v_val;
                    } else {
                        result += attn_scores[t] * static_cast<float>(v[v_idx]);
                    }
                }
                
                size_t out_idx = s * nhead * dv + h * dv + j;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[out_idx] = llaisys::utils::cast<T>(result);
                } else {
                    attn_val[out_idx] = static_cast<T>(result);
                }
            }
        }
    }
}

void self_attention_cpu(std::byte *attn_val, const std::byte *q, const std::byte *k,
                        const std::byte *v, float scale, llaisysDataType_t dtype,
                        size_t seqlen, size_t nhead, size_t nkvhead, size_t d,
                        size_t dv, size_t total_len) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_cpu_(reinterpret_cast<float *>(attn_val),
                                   reinterpret_cast<const float *>(q),
                                   reinterpret_cast<const float *>(k),
                                   reinterpret_cast<const float *>(v),
                                   scale, seqlen, nhead, nkvhead, d, dv, total_len);
    case LLAISYS_DTYPE_F16:
        return self_attention_cpu_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                                   reinterpret_cast<const llaisys::fp16_t *>(q),
                                   reinterpret_cast<const llaisys::fp16_t *>(k),
                                   reinterpret_cast<const llaisys::fp16_t *>(v),
                                   scale, seqlen, nhead, nkvhead, d, dv, total_len);
    case LLAISYS_DTYPE_BF16:
        return self_attention_cpu_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                                   reinterpret_cast<const llaisys::bf16_t *>(q),
                                   reinterpret_cast<const llaisys::bf16_t *>(k),
                                   reinterpret_cast<const llaisys::bf16_t *>(v),
                                   scale, seqlen, nhead, nkvhead, d, dv, total_len);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // Validate inputs
    CHECK_ARGUMENT(q->ndim() == 3, "self_attention: q must be 3-D tensor [seqlen, nhead, d]");
    CHECK_ARGUMENT(k->ndim() == 3, "self_attention: k must be 3-D tensor [total_len, nkvhead, d]");
    CHECK_ARGUMENT(v->ndim() == 3, "self_attention: v must be 3-D tensor [total_len, nkvhead, dv]");
    CHECK_ARGUMENT(attn_val->ndim() == 3, "self_attention: attn_val must be 3-D tensor [seqlen, nhead, dv]");
    
    CHECK_ARGUMENT(attn_val->dtype() == q->dtype(), "self_attention: attn_val and q must have same dtype");
    CHECK_ARGUMENT(q->dtype() == k->dtype(), "self_attention: q and k must have same dtype");
    CHECK_ARGUMENT(q->dtype() == v->dtype(), "self_attention: q and v must have same dtype");
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];
    
    CHECK_ARGUMENT(k->shape()[1] == v->shape()[1], "self_attention: k and v must have same nkvhead");
    CHECK_ARGUMENT(k->shape()[2] == d, "self_attention: k shape[2] must match q shape[2]");
    CHECK_ARGUMENT(v->shape()[0] == total_len, "self_attention: v shape[0] must match k shape[0]");
    CHECK_ARGUMENT(nhead % nkvhead == 0, "self_attention: nhead must be divisible by nkvhead (GQA)");
    CHECK_ARGUMENT(total_len >= seqlen, "self_attention: total_len must be >= seqlen");
    
    CHECK_ARGUMENT(attn_val->shape()[0] == seqlen, "self_attention: attn_val shape[0] must match seqlen");
    CHECK_ARGUMENT(attn_val->shape()[1] == nhead, "self_attention: attn_val shape[1] must match nhead");
    CHECK_ARGUMENT(attn_val->shape()[2] == dv, "self_attention: attn_val shape[2] must match dv");
    
    // Always support CPU calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return self_attention_cpu(attn_val->data(), q->data(), k->data(), v->data(),
                                 scale, attn_val->dtype(), seqlen, nhead, nkvhead, d, dv, total_len);
    }
    
    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return self_attention_cpu(attn_val->data(), q->data(), k->data(), v->data(),
                                 scale, attn_val->dtype(), seqlen, nhead, nkvhead, d, dv, total_len);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
