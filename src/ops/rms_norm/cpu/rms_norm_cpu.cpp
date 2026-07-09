#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, 
               size_t batch_size, size_t feature_dim, float eps) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>)
                    for(size_t i = 0 ; i < batch_size ; i++){
                        float sum = 0.0f;
                        auto base = in + i * feature_dim ;
                        auto baseout = out +i *feature_dim ; 
                        for(size_t j = 0 ; j < feature_dim ;j++){
                            float u = llaisys::utils::cast<float>(base[j]);
                            sum += u*u;
                        }
                        float factor = std::sqrt(sum / static_cast<float>(feature_dim) +eps);
                        for(size_t j = 0 ; j < feature_dim ;j++){
                            float u = llaisys::utils::cast<float>(base[j]);
                            baseout[j] = llaisys::utils::cast<T>(  llaisys::utils::cast<float>(weight[j])*u / factor);
                        }
                    }else{
                        for(size_t i = 0 ; i < batch_size ; i++){
                            float sum = 0.0f;
                            auto base = in + i * feature_dim ;
                            auto baseout = out +i *feature_dim ; 
                            for(size_t j = 0 ; j < feature_dim ;j++){
                                float u = static_cast<float>(base[j]);
                                sum += u*u;
                            }
                            float factor = std::sqrt(sum / static_cast<float>(feature_dim) +eps);
                            for(size_t j = 0 ; j < feature_dim ;j++){
                                float u = static_cast<float>(base[j]);
                                baseout[j] = static_cast<T>(static_cast<float>(weight[j])*u / factor );
                            }
                        }            
                }

}
namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
              llaisysDataType_t type, size_t batch_size, size_t feature_dim, float eps) {
            switch (type) {
                case LLAISYS_DTYPE_F32:
                    return rms_norm_(
                    reinterpret_cast<float *>(out),
                    reinterpret_cast<const float *>(in),
                    reinterpret_cast<const float *>(weight),
                    batch_size, feature_dim, eps
                    );
                case LLAISYS_DTYPE_BF16:
                    return rms_norm_(
                    reinterpret_cast<llaisys::bf16_t *>(out),
                    reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight),
                    batch_size, feature_dim, eps
                    );
                case LLAISYS_DTYPE_F16:
                    return rms_norm_(
                    reinterpret_cast<llaisys::fp16_t *>(out),
                    reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight),
                    batch_size, feature_dim, eps
                    );
                default:
                    EXCEPTION_UNSUPPORTED_DATATYPE(type);
            }  
        }
}