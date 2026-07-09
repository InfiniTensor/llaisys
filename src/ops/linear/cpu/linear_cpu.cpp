#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
static void linear_(T *out, const T *in, const T *W, const T *bias, 
    size_t batch, size_t in_dim, size_t out_dim)
{
    for(size_t i = 0; i < batch; ++i)
    {
        const T *in_ = in + i*in_dim; 
        T *out_ = out + i*out_dim;
        for(size_t j = 0; j < out_dim; ++j)
        {
            const T *weight_ = W + j*in_dim;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> 
                || std::is_same_v<T, llaisys::fp16_t>)
            {
                float acc = 0.0f;
                for(size_t k = 0; k < in_dim; ++k)
                {
                    acc += llaisys::utils::cast<float>(in_[k]) * 
                    llaisys::utils::cast<float>(weight_[k]);
                }
                if(bias)
                    acc += llaisys::utils::cast<float>(bias[j]);
                out_[j] = llaisys::utils::cast<T>(acc);
            }
            else
            {
                T acc = static_cast<T>(0);
                for(size_t k = 0; k < in_dim; ++k)
                {
                    acc += in_[k] * weight_[k];
                }
                if(bias)
                    acc += bias[j];
                out_[j] = acc;
            }
        }
    }

}
namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *W, 
    const std::byte *bias, llaisysDataType_t type, size_t batch, 
    size_t in_dim, size_t out_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(reinterpret_cast<float *>(out), 
        reinterpret_cast<const float *>(in), 
        reinterpret_cast<const float *>(W), 
        reinterpret_cast<const float *>(bias), 
        batch, in_dim, out_dim);
    case LLAISYS_DTYPE_BF16:
        return linear_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), 
        reinterpret_cast<const llaisys::bf16_t *>(in), 
        reinterpret_cast<const llaisys::bf16_t *>(W), 
        reinterpret_cast<const llaisys::bf16_t *>(bias), 
        batch, in_dim, out_dim);  
    case LLAISYS_DTYPE_F16: 
        return linear_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), 
        reinterpret_cast<const llaisys::fp16_t *>(in), 
        reinterpret_cast<const llaisys::fp16_t *>(W), 
        reinterpret_cast<const llaisys::fp16_t *>(bias), 
        batch, in_dim, out_dim);  
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
