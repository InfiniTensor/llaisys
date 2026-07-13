#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath> 

namespace {
    template <typename T>
    void rms_norm_(T *out,const T* input,const T* weight,size_t rows,size_t cols,float eps)
    {
        for(size_t i=0;i<rows;++i)
        {
            float sum_sq = 0.0f;
           for (size_t j = 0; j < cols; ++j) {
            float val = llaisys::utils::cast<float>(input[i * cols + j]);
            sum_sq += val * val;
            }
            float mean_sq= sum_sq / static_cast<float>(cols);
            float inv_rms=1.0f / std::sqrt(mean_sq + eps);

          for (size_t j = 0; j < cols; ++j) {
            float val = llaisys::utils::cast<float>(input[i * cols + j]);
            float w   = llaisys::utils::cast<float>(weight[j]);
            
            float result = val * inv_rms * w;
            
            out[i * cols + j] = llaisys::utils::cast<T>(result);
        }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *input, const std::byte *weight,
              llaisysDataType_t dtype, size_t rows, size_t cols, float eps) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return rms_norm_(reinterpret_cast<float*>(out),
                             reinterpret_cast<const float*>(input),
                             reinterpret_cast<const float*>(weight),
                             rows, cols, eps);
        case LLAISYS_DTYPE_F16:
            return rms_norm_(reinterpret_cast<llaisys::fp16_t*>(out),
                             reinterpret_cast<const llaisys::fp16_t*>(input),
                             reinterpret_cast<const llaisys::fp16_t*>(weight),
                             rows, cols, eps);
        case LLAISYS_DTYPE_BF16:
            return rms_norm_(reinterpret_cast<llaisys::bf16_t*>(out),
                             reinterpret_cast<const llaisys::bf16_t*>(input),
                             reinterpret_cast<const llaisys::bf16_t*>(weight),
                             rows, cols, eps);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu