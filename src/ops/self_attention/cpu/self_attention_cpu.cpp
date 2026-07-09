#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <algorithm>

namespace llaisys::ops::cpu {

template <typename T>
void softmax(T* data, size_t size) {
    T max_val = data[0];
    for (size_t i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    
    T sum = 0;
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<T>(std::exp(static_cast<float>(data[i] - max_val)));
        sum += data[i];
    }
    
    for (size_t i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

template <>
void softmax<llaisys::fp16_t>(llaisys::fp16_t* data, size_t size) {
    float max_val = llaisys::utils::cast<float>(data[0]);
    for (size_t i = 1; i < size; i++) {
        float val = llaisys::utils::cast<float>(data[i]);
        if (val > max_val) {
            max_val = val;
        }
    }
    
    float sum = 0;
    for (size_t i = 0; i < size; i++) {
        float val = llaisys::utils::cast<float>(data[i]);
        data[i] = llaisys::utils::cast<llaisys::fp16_t>(std::exp(val - max_val));
        sum += llaisys::utils::cast<float>(data[i]);
    }
    
    for (size_t i = 0; i < size; i++) {
        float val = llaisys::utils::cast<float>(data[i]);
        data[i] = llaisys::utils::cast<llaisys::fp16_t>(val / sum);
    }
}

template <>
void softmax<llaisys::bf16_t>(llaisys::bf16_t* data, size_t size) {
    float max_val = llaisys::utils::cast<float>(data[0]);
    for (size_t i = 1; i < size; i++) {
        float val = llaisys::utils::cast<float>(data[i]);
        if (val > max_val) {
            max_val = val;
        }
    }
    
    float sum = 0;
    for (size_t i = 0; i < size; i++) {
        float val = llaisys::utils::cast<float>(data[i]);
        data[i] = llaisys::utils::cast<llaisys::bf16_t>(std::exp(val - max_val));
        sum += llaisys::utils::cast<float>(data[i]);
    }
    
    for (size_t i = 0; i < size; i++) {
        float val = llaisys::utils::cast<float>(data[i]);
        data[i] = llaisys::utils::cast<llaisys::bf16_t>(val / sum);
    }
}

template <typename T>
void self_attention_impl(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, size_t seqlen, size_t nhead, size_t nkhead, size_t d, size_t total_len) {
    const T* q_data = reinterpret_cast<const T*>(q);
    const T* k_data = reinterpret_cast<const T*>(k);
    const T* v_data = reinterpret_cast<const T*>(v);
    T* attn_val_data = reinterpret_cast<T*>(attn_val);
    
    size_t repeats = nhead / nkhead;
    
    T* k_expanded = new T[total_len * nhead * d];
    T* v_expanded = new T[total_len * nhead * d];
    
    for (size_t i = 0; i < total_len; i++) {
        for (size_t j = 0; j < nhead; j++) {
            size_t kv_head = j / repeats;
            for (size_t k_idx = 0; k_idx < d; k_idx++) {
                size_t src_idx = i * nkhead * d + kv_head * d + k_idx;
                size_t dst_idx = i * nhead * d + j * d + k_idx;
                k_expanded[dst_idx] = k_data[src_idx];
                v_expanded[dst_idx] = v_data[src_idx];
            }
        }
    }
    
    T* attn_scores = new T[nhead * seqlen * total_len];
    
    for (size_t j = 0; j < nhead; j++) {
        for (size_t i = 0; i < seqlen; i++) {
            for (size_t t = 0; t < total_len; t++) {
                float score = 0.0f;
                for (size_t k_idx = 0; k_idx < d; k_idx++) {
                    size_t q_idx = i * nhead * d + j * d + k_idx;
                    size_t k_idx_local = t * nhead * d + j * d + k_idx;
                    score += llaisys::utils::cast<float>(q_data[q_idx]) * llaisys::utils::cast<float>(k_expanded[k_idx_local]);
                }
                score *= scale;
                
                size_t mask_threshold = (total_len > seqlen) ? (i + total_len - seqlen) : i;
                if (t > mask_threshold) {
                    score = -1e9f;
                }
                
                attn_scores[j * seqlen * total_len + i * total_len + t] = llaisys::utils::cast<T>(score);
            }
            softmax(&attn_scores[j * seqlen * total_len + i * total_len], total_len);
        }
    }
    
    for (size_t j = 0; j < nhead; j++) {
        for (size_t i = 0; i < seqlen; i++) {
            for (size_t d_idx = 0; d_idx < d; d_idx++) {
                float val = 0.0f;
                for (size_t t = 0; t < total_len; t++) {
                    float attn_weight = llaisys::utils::cast<float>(attn_scores[j * seqlen * total_len + i * total_len + t]);
                    size_t v_idx = t * nhead * d + j * d + d_idx;
                    val += attn_weight * llaisys::utils::cast<float>(v_expanded[v_idx]);
                }
                size_t out_idx = i * nhead * d + j * d + d_idx;
                attn_val_data[out_idx] = llaisys::utils::cast<T>(val);
            }
        }
    }
    
    delete[] attn_scores;
    delete[] k_expanded;
    delete[] v_expanded;
}

template <>
void self_attention_impl<llaisys::fp16_t>(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, size_t seqlen, size_t nhead, size_t nkhead, size_t d, size_t total_len) {
    const llaisys::fp16_t* q_data = reinterpret_cast<const llaisys::fp16_t*>(q);
    const llaisys::fp16_t* k_data = reinterpret_cast<const llaisys::fp16_t*>(k);
    const llaisys::fp16_t* v_data = reinterpret_cast<const llaisys::fp16_t*>(v);
    llaisys::fp16_t* attn_val_data = reinterpret_cast<llaisys::fp16_t*>(attn_val);
    
    size_t repeats = nhead / nkhead;
    
    llaisys::fp16_t* k_expanded = new llaisys::fp16_t[total_len * nhead * d];
    llaisys::fp16_t* v_expanded = new llaisys::fp16_t[total_len * nhead * d];
    
    for (size_t i = 0; i < total_len; i++) {
        for (size_t j = 0; j < nhead; j++) {
            size_t kv_head = j / repeats;
            for (size_t k_idx = 0; k_idx < d; k_idx++) {
                size_t src_idx = i * nkhead * d + kv_head * d + k_idx;
                size_t dst_idx = i * nhead * d + j * d + k_idx;
                k_expanded[dst_idx] = k_data[src_idx];
                v_expanded[dst_idx] = v_data[src_idx];
            }
        }
    }
    
    llaisys::fp16_t* attn_scores = new llaisys::fp16_t[nhead * seqlen * total_len];
    
    for (size_t j = 0; j < nhead; j++) {
        for (size_t i = 0; i < seqlen; i++) {
            for (size_t t = 0; t < total_len; t++) {
                float score = 0.0f;
                for (size_t k_idx = 0; k_idx < d; k_idx++) {
                    size_t q_idx = i * nhead * d + j * d + k_idx;
                    size_t k_idx_local = t * nhead * d + j * d + k_idx;
                    score += llaisys::utils::cast<float>(q_data[q_idx]) * llaisys::utils::cast<float>(k_expanded[k_idx_local]);
                }
                score *= scale;
                
                size_t mask_threshold = (total_len > seqlen) ? (i + total_len - seqlen) : i;
                if (t > mask_threshold) {
                    score = -1e9f;
                }
                
                attn_scores[j * seqlen * total_len + i * total_len + t] = llaisys::utils::cast<llaisys::fp16_t>(score);
            }
            softmax(&attn_scores[j * seqlen * total_len + i * total_len], total_len);
        }
    }
    
    for (size_t j = 0; j < nhead; j++) {
        for (size_t i = 0; i < seqlen; i++) {
            for (size_t d_idx = 0; d_idx < d; d_idx++) {
                float val = 0.0f;
                for (size_t t = 0; t < total_len; t++) {
                    float attn_weight = llaisys::utils::cast<float>(attn_scores[j * seqlen * total_len + i * total_len + t]);
                    size_t v_idx = t * nhead * d + j * d + d_idx;
                    val += attn_weight * llaisys::utils::cast<float>(v_expanded[v_idx]);
                }
                size_t out_idx = i * nhead * d + j * d + d_idx;
                attn_val_data[out_idx] = llaisys::utils::cast<llaisys::fp16_t>(val);
            }
        }
    }
    
    delete[] attn_scores;
    delete[] k_expanded;
    delete[] v_expanded;
}

template <>
void self_attention_impl<llaisys::bf16_t>(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, size_t seqlen, size_t nhead, size_t nkhead, size_t d, size_t total_len) {
    const llaisys::bf16_t* q_data = reinterpret_cast<const llaisys::bf16_t*>(q);
    const llaisys::bf16_t* k_data = reinterpret_cast<const llaisys::bf16_t*>(k);
    const llaisys::bf16_t* v_data = reinterpret_cast<const llaisys::bf16_t*>(v);
    llaisys::bf16_t* attn_val_data = reinterpret_cast<llaisys::bf16_t*>(attn_val);
    
    size_t repeats = nhead / nkhead;
    
    llaisys::bf16_t* k_expanded = new llaisys::bf16_t[total_len * nhead * d];
    llaisys::bf16_t* v_expanded = new llaisys::bf16_t[total_len * nhead * d];
    
    for (size_t i = 0; i < total_len; i++) {
        for (size_t j = 0; j < nhead; j++) {
            size_t kv_head = j / repeats;
            for (size_t k_idx = 0; k_idx < d; k_idx++) {
                size_t src_idx = i * nkhead * d + kv_head * d + k_idx;
                size_t dst_idx = i * nhead * d + j * d + k_idx;
                k_expanded[dst_idx] = k_data[src_idx];
                v_expanded[dst_idx] = v_data[src_idx];
            }
        }
    }
    
    llaisys::bf16_t* attn_scores = new llaisys::bf16_t[nhead * seqlen * total_len];
    
    for (size_t j = 0; j < nhead; j++) {
        for (size_t i = 0; i < seqlen; i++) {
            for (size_t t = 0; t < total_len; t++) {
                float score = 0.0f;
                for (size_t k_idx = 0; k_idx < d; k_idx++) {
                    size_t q_idx = i * nhead * d + j * d + k_idx;
                    size_t k_idx_local = t * nhead * d + j * d + k_idx;
                    score += llaisys::utils::cast<float>(q_data[q_idx]) * llaisys::utils::cast<float>(k_expanded[k_idx_local]);
                }
                score *= scale;
                
                size_t mask_threshold = (total_len > seqlen) ? (i + total_len - seqlen) : i;
                if (t > mask_threshold) {
                    score = -1e9f;
                }
                
                attn_scores[j * seqlen * total_len + i * total_len + t] = llaisys::utils::cast<llaisys::bf16_t>(score);
            }
            softmax(&attn_scores[j * seqlen * total_len + i * total_len], total_len);
        }
    }
    
    for (size_t j = 0; j < nhead; j++) {
        for (size_t i = 0; i < seqlen; i++) {
            for (size_t d_idx = 0; d_idx < d; d_idx++) {
                float val = 0.0f;
                for (size_t t = 0; t < total_len; t++) {
                    float attn_weight = llaisys::utils::cast<float>(attn_scores[j * seqlen * total_len + i * total_len + t]);
                    size_t v_idx = t * nhead * d + j * d + d_idx;
                    val += attn_weight * llaisys::utils::cast<float>(v_expanded[v_idx]);
                }
                size_t out_idx = i * nhead * d + j * d + d_idx;
                attn_val_data[out_idx] = llaisys::utils::cast<llaisys::bf16_t>(val);
            }
        }
    }
    
    delete[] attn_scores;
    delete[] k_expanded;
    delete[] v_expanded;
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t num_kv_heads, size_t d, size_t total_len) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_impl<float>(attn_val, q, k, v, scale, seqlen, nhead, num_kv_heads, d, total_len);
    case LLAISYS_DTYPE_F64:
        return self_attention_impl<double>(attn_val, q, k, v, scale, seqlen, nhead, num_kv_heads, d, total_len);
    case LLAISYS_DTYPE_F16:
        return self_attention_impl<llaisys::fp16_t>(attn_val, q, k, v, scale, seqlen, nhead, num_kv_heads, d, total_len);
    case LLAISYS_DTYPE_BF16:
        return self_attention_impl<llaisys::bf16_t>(attn_val, q, k, v, scale, seqlen, nhead, num_kv_heads, d, total_len);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
