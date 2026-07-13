#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>

namespace { 

template <typename T>
void rope_(T *out, const T *input, const int64_t *pos_ids,
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    
    size_t dim_half = head_dim / 2;

    std::vector<float> cos_cache(dim_half);
    std::vector<float> sin_cache(dim_half);

    for (size_t s = 0; s < seq_len; ++s) {
        int64_t pos = pos_ids[s];

        for (size_t j = 0; j < dim_half; ++j) {
            float freq_expon = static_cast<float>(2 * j) / static_cast<float>(head_dim);
            float freq = static_cast<float>(pos) / std::pow(theta, freq_expon);
            
            cos_cache[j] = std::cos(freq);
            sin_cache[j] = std::sin(freq);
        }

        for (size_t h = 0; h < n_heads; ++h) {

            size_t offset = s * n_heads * head_dim + h * head_dim;
            
            const T* src_vec = input + offset;
            T* dst_vec = out + offset;

            for (size_t j = 0; j < dim_half; ++j) {

                float val_a = llaisys::utils::cast<float>(src_vec[j]);
                float val_b = llaisys::utils::cast<float>(src_vec[j + dim_half]);

                float cos_val = cos_cache[j];
                float sin_val = sin_cache[j];

                float res_a = val_a * cos_val - val_b * sin_val;
                float res_b = val_b * cos_val + val_a * sin_val;

                dst_vec[j]            = llaisys::utils::cast<T>(res_a);
                dst_vec[j + dim_half] = llaisys::utils::cast<T>(res_b);
            }
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void rope(std::byte *out, const std::byte *input, const std::byte *pos_ids,
          llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t head_dim,
          float theta) {
    
    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids);

    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return rope_(reinterpret_cast<float*>(out),
                         reinterpret_cast<const float*>(input),
                         pos_ptr, seq_len, n_heads, head_dim, theta);
        case LLAISYS_DTYPE_F16:
            return rope_(reinterpret_cast<llaisys::fp16_t*>(out),
                         reinterpret_cast<const llaisys::fp16_t*>(input),
                         pos_ptr, seq_len, n_heads, head_dim, theta);
        case LLAISYS_DTYPE_BF16:
            return rope_(reinterpret_cast<llaisys::bf16_t*>(out),
                         reinterpret_cast<const llaisys::bf16_t*>(input),
                         pos_ptr, seq_len, n_heads, head_dim, theta);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu