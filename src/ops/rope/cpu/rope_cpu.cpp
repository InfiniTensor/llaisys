#include "rope_cpu.hpp"
#include <cmath>
#include "../../../utils.hpp"

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, const float theta,
    size_t seq_len, size_t n_heads, size_t head_dim)
{
    //for each s in seq_len
    for(size_t s = 0; s < seq_len; ++s)
    {
        //for each h in n_heads
        for(size_t h = 0; h < n_heads; ++h)
        {
            //for each tensor in head_dim
            for(size_t j = 0; j < head_dim/2; ++j)
            {
                size_t p = pos_ids[s];
                // cal angle
                float angle = static_cast<float>(p) / std::pow(theta, 2.0f*j / head_dim);
                // cal the base
                const T *in_ = in + (s*n_heads + h) * head_dim;
                T *out_ = out + (s*n_heads + h) * head_dim;
                //cal sin & cos val
                float cosine = std::cos(angle);
                float sine = std::sin(angle);
                //find a and b
                T a = in_[j];
                T b = in_[j + head_dim/2];
                //rope
                if constexpr(std::is_same_v<T, llaisys::bf16_t> 
                    || std::is_same_v<T, llaisys::fp16_t>)
                {
                    out_[j] = llaisys::utils::cast<T>(
                        llaisys::utils::cast<float>(a) * cosine - 
                        llaisys::utils::cast<float>(b) * sine
                    );
                    out_[j + head_dim/2] = llaisys::utils::cast<T>(
                        llaisys::utils::cast<float>(a) * sine + 
                        llaisys::utils::cast<float>(b) * cosine
                    );
                }
                else
                {
                    out_[j] = a * cosine - b * sine;
                    out_[j + head_dim/2] = a * sine + b * cosine;
                }
            }
        }
    }
}
namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, 
    const std::byte *pos_ids, const float theta, 
    llaisysDataType_t type, 
    size_t seq_len, size_t n_heads, size_t head_dim) {  
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(reinterpret_cast<float *>(out), 
        reinterpret_cast<const float *>(in), 
        reinterpret_cast<const int64_t *>(pos_ids), theta,
        seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), 
        reinterpret_cast<const llaisys::bf16_t *>(in), 
        reinterpret_cast<const int64_t *>(pos_ids), theta,
        seq_len, n_heads, head_dim);  
    case LLAISYS_DTYPE_F16: 
        return rope_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), 
        reinterpret_cast<const llaisys::fp16_t *>(in), 
        reinterpret_cast<const int64_t *>(pos_ids), theta,
        seq_len, n_heads, head_dim);    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
