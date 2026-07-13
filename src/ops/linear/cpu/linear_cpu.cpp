#include "linear_cpu.hpp"
#include "../../../utils.hpp"

namespace{
    template<typename T>
    void linear_(T* out,const T* input,const T* weight,const T* bias,size_t M,size_t K,size_t N)
    {
        for(size_t m=0;m<M;++m)
        {
            for(size_t n=0;n<N;++n)
            {
                float sum=0.0f;
                for(size_t k=0;k<K;++k)
                {
                    float x_val=llaisys::utils::cast<float>(input[m*K+k]);
                    float w_val=llaisys::utils::cast<float>(weight[n*K+k]);
                    sum+=x_val*w_val;
                }
                if(bias)
                {
                    sum+=llaisys::utils::cast<float>(bias[n]);
                }
                out[m*N+n]=llaisys::utils::cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu
{
    void linear(std::byte *out, const std::byte *input, const std::byte *weight, const std::byte *bias,llaisysDataType_t dtype, size_t M, size_t K, size_t N)
    {
       switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return linear_(reinterpret_cast<float*>(out), 
                           reinterpret_cast<const float*>(input),
                           reinterpret_cast<const float*>(weight),
                           reinterpret_cast<const float*>(bias), // bias 为空时这里也是 nullptr
                           M, K, N);
        case LLAISYS_DTYPE_F16:
            return linear_(reinterpret_cast<llaisys::fp16_t*>(out), 
                           reinterpret_cast<const llaisys::fp16_t*>(input),
                           reinterpret_cast<const llaisys::fp16_t*>(weight),
                           reinterpret_cast<const llaisys::fp16_t*>(bias),
                           M, K, N);
        case LLAISYS_DTYPE_BF16:
            return linear_(reinterpret_cast<llaisys::bf16_t*>(out), 
                           reinterpret_cast<const llaisys::bf16_t*>(input),
                           reinterpret_cast<const llaisys::bf16_t*>(weight),
                           reinterpret_cast<const llaisys::bf16_t*>(bias),
                           M, K, N);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    }
}// namespace llaisys::ops::cpu