#include "op.hpp"
#include<cmath>
#include<cstring>
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    auto get_float_at=[&](const std::byte* data,size_t i,size_t j,size_t stride_i,size_t stride_j)->float{
        const std::byte* ptr=data+(i*stride_i+j*stride_j)*utils::dsize(in->dtype());
        switch(in->dtype()){
            case LLAISYS_DTYPE_F32:{
                float val;
                memcpy(&val,ptr,sizeof(float));
                return val;
            }
            case LLAISYS_DTYPE_F16:{
                fp16_t val;
                memcpy(&val, ptr, sizeof(fp16_t));
                return utils::cast<float>(val);
            }
            case LLAISYS_DTYPE_BF16:{
                bf16_t val;
                memcpy(&val, ptr, sizeof(bf16_t));
                return utils::cast<float>(val);
            }
            default:EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
        }
    };
    auto set_float_at=[&](std::byte* data,size_t i,size_t j,size_t stride_i,size_t stride_j,float val){
        std::byte* ptr = data + (i * stride_i + j * stride_j) * utils::dsize(in->dtype());
        switch(in->dtype()){
            case LLAISYS_DTYPE_F32:{
                memcpy(ptr,&val,sizeof(float));
                break;
            }
            case LLAISYS_DTYPE_F16:{
                fp16_t h=utils::cast<fp16_t>(val);
                memcpy(ptr,&h,sizeof(fp16_t));
                break;
            }
            case LLAISYS_DTYPE_BF16:{
                bf16_t b=utils::cast<bf16_t>(val);
                memcpy(ptr,&b,sizeof(bf16_t));
                break;
            }
            default:EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    };
    auto get_int_at=[&](const std::byte* data,size_t i,size_t j,size_t stride_i,size_t stride_j)->int64_t{
        const std::byte* ptr=data+(i*stride_i+j*stride_j)*utils::dsize(pos_ids->dtype());
        switch(pos_ids->dtype()){
            case LLAISYS_DTYPE_I64:{
                int64_t val;
                memcpy(&val,ptr,sizeof(int64_t));
                return val;
            }
            case LLAISYS_DTYPE_I32:{
                int32_t val;
                memcpy(&val,ptr,sizeof(int32_t));
                return val;
            }
            case LLAISYS_DTYPE_I16:{
                int16_t val;
                memcpy(&val,ptr,sizeof(int16_t));
                return val;
            }
            default:throw std::runtime_error("Unsupported data type: " + std::to_string(pos_ids->dtype()));
        }
    };
    for(size_t i=0;i<in->shape()[0];i++){
        int64_t p_i=get_int_at(pos_ids->data(),i,0,pos_ids->strides()[0],pos_ids->strides()[1]);
        for(size_t h=0;h<in->shape()[1];h++){
            for(size_t j=0;j<in->shape()[2]/2;j++){
                float phi=p_i/powf(theta,2.0f*j/in->shape()[2]);
                float a_j=get_float_at(in->data(),i,h*in->shape()[2]+j,in->strides()[0],in->strides()[1]);
                float b_j=get_float_at(in->data(),i,h*in->shape()[2]+j+in->shape()[2]/2,in->strides()[0],in->strides()[1]);
                float a_prime=a_j*cosf(phi)-b_j*sinf(phi);
                float b_prime=b_j*cosf(phi)+a_j*sinf(phi);
                set_float_at(out->data(),i,h*in->shape()[2]+j,out->strides()[0],out->strides()[1],a_prime);
                set_float_at(out->data(),i,h*in->shape()[2]+j+in->shape()[2]/2,out->strides()[0],out->strides()[1],b_prime);
            }
        }
    }
}
} // namespace llaisys::ops