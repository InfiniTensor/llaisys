#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cassert>


template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
                auto half_dim = head_dim / 2;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                for(size_t s = 0 ; s < seq_len ; s++){

                    auto position = pos_ids[s];

                    for(size_t h = 0 ; h < n_heads ; h++){

                        auto in_a = in + s* n_heads * head_dim + h * head_dim ;
                        auto in_b = in + s* n_heads * head_dim + h * head_dim + half_dim;
                        auto out_a = out + s* n_heads * head_dim + h * head_dim;
                        auto out_b = out + s* n_heads * head_dim + h * head_dim + half_dim;

                        for(size_t d = 0 ; d < half_dim ; d++){

                            float phi = position /(std::pow(theta , 2.0f * static_cast<float>(d) /static_cast<float>(head_dim) )) ; 
                            float cosval = std::cos(phi);
                            float sinval = std::sin(phi);
                            out_a[d] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in_a[d]) *cosval - llaisys::utils::cast<float>(in_b[d]) * sinval);
                            out_b[d] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in_b[d]) *cosval + llaisys::utils::cast<float>(in_a[d]) * sinval);
                        }
                    }
                }
           } else{
                for(size_t s = 0 ; s < seq_len ; s++){

                    auto position = pos_ids[s];

                    for(size_t h = 0 ; h < n_heads ; h++){

                        auto in_a = in + s* n_heads * head_dim + h * head_dim ;
                        auto in_b = in + s* n_heads * head_dim + h * head_dim + half_dim;
                        auto out_a = out + s* n_heads * head_dim + h * head_dim;
                        auto out_b = out + s* n_heads * head_dim + h * head_dim + half_dim;

                        for(size_t d = 0 ; d < half_dim ; d++){

                            float phi = position /(std::pow(theta , 2.0f * static_cast<float>(d) /static_cast<float>(head_dim) )) ; 
                            float cosval = std::cos(phi);
                            float sinval = std::sin(phi);
                            out_a[d] = static_cast<T>(static_cast<float>(in_a[d]) *cosval - static_cast<float>(in_b[d]) * sinval);
                            out_b[d] = static_cast<T>(static_cast<float>(in_b[d]) *cosval + static_cast<float>(in_a[d]) * sinval);
                        }
                    }
                }
           }
}
namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
          llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, head_dim, theta
        );
    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, head_dim, theta
        );
    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, head_dim, theta
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}