#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>

namespace llaisys::ops::cpu {

// Helper function for softmax with numerical stability
void softmax(float *x, size_t size) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    if (max_val == -std::numeric_limits<float>::infinity()) {
        for (size_t i = 0; i < size; ++i) {
            x[i] = 1.0f / size;
        }
        return;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    for (size_t i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

template <typename T>
void self_attention_(std::byte *attn_val_bytes, const std::byte *q_bytes, const std::byte *k_bytes, const std::byte *v_bytes, float scale, size_t seq_len, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    auto attn_val = reinterpret_cast<T *>(attn_val_bytes);
    auto q = reinterpret_cast<const T *>(q_bytes);
    auto k = reinterpret_cast<const T *>(k_bytes);
    auto v = reinterpret_cast<const T *>(v_bytes);

    // Calculate repeats for Grouped Query Attention (GQA)
    size_t num_repeats = nhead / nkvhead;
    
    // Outer loop over heads
    for (size_t h = 0; h < nhead; ++h) {
        // Map current query head `h` to the corresponding key/value head `kvh`
        size_t kvh = h / num_repeats;
        
        // Loop over Sequence Length (Time)
        for (size_t i = 0; i < seq_len; ++i) {
            float* attn_scores = new float[total_len];
            
            // 1. Compute Q @ K^T
            for (size_t j = 0; j < total_len; ++j) {
                double dot_product = 0.0;
                for (size_t k_idx = 0; k_idx < d; ++k_idx) {
                    // Correct Indexing for [seq_len, nhead, d] layout
                    // Stride(i) = nhead * d
                    // Stride(h) = d
                    size_t q_pos = i * nhead * d + h * d + k_idx;
                    double q_val = static_cast<double>(llaisys::utils::cast<float>(q[q_pos]));
                    
                    // Correct Indexing for [total_len, nkvhead, d] layout
                    // Stride(j) = nkvhead * d
                    // Stride(kvh) = d
                    size_t k_pos = j * nkvhead * d + kvh * d + k_idx;
                    double k_val = static_cast<double>(llaisys::utils::cast<float>(k[k_pos]));
                    
                    dot_product += q_val * k_val;
                }
                
                attn_scores[j] = static_cast<float>(dot_product * static_cast<double>(scale));
            }
            
            // 2. Apply Causal Mask
            size_t diagonal = total_len - seq_len;
            for (size_t j = 0; j < total_len; ++j) {
                if (j > i + diagonal) {
                    attn_scores[j] = -std::numeric_limits<float>::infinity();
                }
            }
            
            // 3. Softmax
            softmax(attn_scores, total_len);
            
            // 4. Compute Weighted Sum (Scores @ V)
            for (size_t v_idx = 0; v_idx < dv; ++v_idx) {
                double weighted_sum = 0.0;
                for (size_t j = 0; j < total_len; ++j) {
                    // Correct Indexing for [total_len, nkvhead, dv] layout
                    size_t v_pos = j * nkvhead * dv + kvh * dv + v_idx;
                    double v_val = static_cast<double>(llaisys::utils::cast<float>(v[v_pos]));
                    
                    weighted_sum += static_cast<double>(attn_scores[j]) * v_val;
                }
                
                // Output layout is [seq_len, nhead, dv]
                size_t out_pos = i * nhead * dv + h * dv + v_idx;
                attn_val[out_pos] = llaisys::utils::cast<T>(static_cast<float>(weighted_sum));
            }
            
            delete[] attn_scores;
        }
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, float scale, size_t seq_len, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(attn_val, q, k, v, scale, seq_len, total_len, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<llaisys::bf16_t>(attn_val, q, k, v, scale, seq_len, total_len, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_<llaisys::fp16_t>(attn_val, q, k, v, scale, seq_len, total_len, nhead, nkvhead, d, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu