#include "rms_norm_cpu.hpp"
#include <cmath>
#include "../../../utils.hpp"

template <typename T>
static void rms_norm_(T *out, const T *in, const T *W, 
    const float eps, size_t batch, size_t in_dim)
{
    //for each row in in_tensor
    for(size_t i = 0; i < batch; ++i)
    {
        const T *in_ = in + i*in_dim;
        T *out_ = out + i*in_dim;
        float std_dev = 0.0f;
        //first calculate the standard deviation of each row
        if constexpr (std::is_same_v<T, llaisys::bf16_t> 
            || std::is_same_v<T, llaisys::fp16_t>)
        {
            for(size_t j = 0; j < in_dim; ++j)
            {
                std_dev += llaisys::utils::cast<float>(in_[j]) * 
                llaisys::utils::cast<float>(in_[j]);
            }
            std_dev = std::sqrt(std_dev / (int)in_dim + eps);
        }
        else
        {
            for(size_t j = 0; j < in_dim; ++j)
            {
                std_dev += in_[j] * in_[j];
            }
            std_dev = std::sqrt(std_dev / in_dim + eps);
        }
        //then normalize each element in the row
        for(size_t j = 0; j < in_dim; ++j)
        {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> 
                || std::is_same_v<T, llaisys::fp16_t>)
            {
                out_[j] = llaisys::utils::cast<T>(
                    llaisys::utils::cast<float>(in_[j]) * 
                    llaisys::utils::cast<float>(W[j]) / std_dev);
            }
            else
            {
                out_[j] = in_[j] * W[j] / std_dev;
            }
        }
    }

}
namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, 
    const std::byte *W, const float eps, 
    llaisysDataType_t type, 
    size_t batch, size_t in_dim) {  
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(reinterpret_cast<float *>(out), 
        reinterpret_cast<const float *>(in), 
        reinterpret_cast<const float *>(W), 
        eps, batch, in_dim);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), 
        reinterpret_cast<const llaisys::bf16_t *>(in), 
        reinterpret_cast<const llaisys::bf16_t *>(W), 
        eps, batch, in_dim);  
    case LLAISYS_DTYPE_F16: 
        return rms_norm_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), 
        reinterpret_cast<const llaisys::fp16_t *>(in), 
        reinterpret_cast<const llaisys::fp16_t *>(W), 
        eps, batch, in_dim);  
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
