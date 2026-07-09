#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                    size_t seqlen, size_t totlen, size_t nh, size_t nkvh, size_t d, size_t dv, float scale) {
                    auto shared = nh/nkvh ;
                    if constexpr( std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>)
                    for(size_t s = 0 ; s<seqlen ;s++){
                        for(size_t h=0 ; h<nh ; h++){
                            auto kvhead = h/shared ;
                            auto baseq = q + s * nh * d + h *d;
                            auto baseout = attn_val + s * nh*dv+ h*dv;
                            float* attn = new float[totlen];
                            for(size_t t = 0; t<totlen ;t++){
                                auto basek = k + t * nkvh * d + kvhead *d;
                                attn[t]= 0;
                                for(size_t d_ = 0;d_<d ; d_++){
                                    attn[t]+=llaisys::utils::cast<float>(baseq[d_])*llaisys::utils::cast<float>(basek[d_]);
                                }
                                attn[t] *= scale;
                                if(t > s + totlen - seqlen){
                                    attn[t] = -std::numeric_limits<float>::infinity();
                                }
                            }
                            float max_attn = -std::numeric_limits<float>::infinity();
                            for (size_t t= 0; t < totlen; t++) {
                                if (attn[t] > max_attn) {
                                    max_attn = attn[t];
                                }
                            }
                            float sum_exp = 0.0f;
                            for (size_t t = 0; t < totlen; t++) {
                                attn[t] = std::exp(attn[t] - max_attn);
                                sum_exp += attn[t];
                            } 
                            for (size_t t = 0; t < totlen; t++) {
                                attn[t] /= sum_exp;
                            }
                            for (size_t dv_ = 0;dv_ < dv;dv_++){
                                float val = 0;
                                for(size_t t = 0; t < totlen; t++){
                                    auto basev  = v+ t* nkvh * dv + kvhead * dv;
                                    val+=llaisys::utils::cast<float>( basev[dv_])*attn[t];
                                }
                                baseout[dv_]=llaisys::utils::cast<T>(val);
                            }
                            delete[]attn;
                        }
                    }
                    else{
                    for(size_t s = 0 ; s<seqlen ;s++){
                        for(size_t h=0 ; h<nh ; h++){
                            auto kvhead = h/shared ;
                            auto baseq = q + s * nh * d + h *d;
                            auto baseout = attn_val + s * nh*dv+ h*dv;
                            float* attn = new float[totlen];
                            for(size_t t = 0; t<totlen ;t++){
                                auto basek = k + t * nkvh * d + kvhead *d;
                                attn[t]= 0;
                                for(size_t d_ = 0;d_<d ; d_++){
                                    attn[t]+=static_cast<float>(baseq[d_])*static_cast<float>(basek[d_]);
                                }
                                attn[t]*= scale;
                                if(t > s + totlen - seqlen){
                                    attn[t] = -std::numeric_limits<float>::infinity();
                                }
                            }
                            float max_attn = -std::numeric_limits<float>::infinity();
                            for (size_t t= 0; t < totlen; t++) {
                                if (attn[t] > max_attn) {
                                    max_attn = attn[t];
                                }
                            }
                            float sum_exp = 0.0f;
                            for (size_t t = 0; t < totlen; t++) {
                                attn[t] = std::exp(attn[t] - max_attn);
                                sum_exp += attn[t];
                            } 
                            for (size_t t = 0; t < totlen; t++) {
                                attn[t] /= sum_exp;
                            }
                            for (size_t dv_ = 0;dv_ < dv;dv_++){
                                float val = 0;
                                for(size_t t = 0; t < totlen; t++){
                                    auto basev  = v+ t* nkvh * dv + kvhead * dv;
                                    val+=static_cast<float>( basev[dv_])*attn[t];
                                }
                                baseout[dv_]=llaisys::utils::cast<T>(val);
                            }
                            delete[]attn;
                        }
                    }
                }
        }
                

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                   llaisysDataType_t type, size_t seqlen, size_t totlen, size_t nh, size_t nkvh, 
                   size_t d, size_t dv , float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            seqlen, totlen, nh, nkvh, d, dv , scale
        );
    case LLAISYS_DTYPE_BF16:
        return self_attention_(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            seqlen, totlen, nh, nkvh, d, dv , scale
        );
    case LLAISYS_DTYPE_F16:
        return self_attention_(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            seqlen, totlen, nh, nkvh, d, dv , scale
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu