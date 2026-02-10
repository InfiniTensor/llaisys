#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace {

template <typename T>
void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v, float scale, 
                            const std::vector<size_t> &q_shape, 
                            const std::vector<size_t> &k_shape, 
                            const std::vector<size_t> &v_shape) {

    size_t sq = q_shape[0]; 
    size_t nh = q_shape[1];
    size_t d  = q_shape[2];

    size_t sk    = k_shape[0];
    size_t nh_kv = k_shape[1];

    size_t dv    = v_shape[2]; 

    size_t n_rep = nh / nh_kv;

    for (size_t i = 0; i < sq; ++i) {
        
        size_t q_abs_pos = sk - sq + i;

        for (size_t h = 0; h < nh; ++h) {
            size_t h_kv = h / n_rep;

            const T* q_vec = q + (i * nh * d) + (h * d);

            std::vector<float> scores(sk);
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t j = 0; j < sk; ++j) {
                if (j > q_abs_pos) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const T* k_vec = k + (j * nh_kv * d) + (h_kv * d);
                float dot = 0.0f;
                for (size_t l = 0; l < d; ++l) {
                    float val_q = llaisys::utils::cast<float>(q_vec[l]);
                    float val_k = llaisys::utils::cast<float>(k_vec[l]);
                    dot += val_q * val_k;
                }
                
                float score = dot * scale;
                scores[j] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }
            
            float sum_exp = 0.0f;
            for (size_t j = 0; j < sk; ++j) {
                if (scores[j] == -std::numeric_limits<float>::infinity()) {
                    scores[j] = 0.0f;
                } else {
                    float exp_val = std::exp(scores[j] - max_score);
                    scores[j] = exp_val;
                    sum_exp += exp_val;
                }
            }
            
            float inv_sum = 1.0f / (sum_exp + 1e-10f);
            
            std::vector<float> out_accum(dv, 0.0f);

            for (size_t j = 0; j < sk; ++j) {
                float weight = scores[j] * inv_sum;
                
                if (weight < 1e-10f) continue;
                const T* v_vec = v + (j * nh_kv * dv) + (h_kv * dv);

                for (size_t l = 0; l < dv; ++l) {
                    float val_v = llaisys::utils::cast<float>(v_vec[l]);
                    out_accum[l] += weight * val_v;
                }
            }

            T* out_ptr = attn_val + (i * nh * dv) + (h * dv);
            for (size_t l = 0; l < dv; ++l) {
                out_ptr[l] = llaisys::utils::cast<T>(out_accum[l]);
            }
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, llaisysDataType_t dtype, 
                    const std::vector<size_t> &q_shape, 
                    const std::vector<size_t> &k_shape, 
                    const std::vector<size_t> &v_shape) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel(reinterpret_cast<float*>(attn_val), reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k), reinterpret_cast<const float*>(v), scale, q_shape, k_shape, v_shape);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel(reinterpret_cast<llaisys::bf16_t*>(attn_val), reinterpret_cast<const llaisys::bf16_t*>(q), reinterpret_cast<const llaisys::bf16_t*>(k), reinterpret_cast<const llaisys::bf16_t*>(v),  scale, q_shape, k_shape, v_shape);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel(reinterpret_cast<llaisys::fp16_t*>(attn_val), reinterpret_cast<const llaisys::fp16_t*>(q), reinterpret_cast<const llaisys::fp16_t*>(k), reinterpret_cast<const llaisys::fp16_t*>(v), scale, q_shape, k_shape, v_shape);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
