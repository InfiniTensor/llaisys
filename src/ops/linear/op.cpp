#include "op.hpp"
#include <cstring>
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    auto get_float_at=[&](const std::byte* data,size_t elem_offset,llaisysDataType_t dtype)->float{
        const std::byte* ptr=data+elem_offset*utils::dsize(dtype);
        switch(dtype){
            case LLAISYS_DTYPE_F32:{
                float val;std::memcpy(&val,ptr,sizeof(float));return val;
            }
            case LLAISYS_DTYPE_F16: {
                fp16_t val;std::memcpy(&val,ptr,sizeof(fp16_t));return utils::cast<float>(val);
            }
            case LLAISYS_DTYPE_BF16: {
                bf16_t val;std::memcpy(&val,ptr,sizeof(bf16_t));return utils::cast<float>(val);
            }
            default:EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };
    auto set_float_at=[&](std::byte* data,size_t elem_offset,float val,llaisysDataType_t dtype) {
        std::byte* ptr=data+elem_offset*utils::dsize(dtype);
        switch(dtype){
            case LLAISYS_DTYPE_F32:{
                std::memcpy(ptr,&val,sizeof(float));break;
            }
            case LLAISYS_DTYPE_F16: {
                fp16_t h=utils::cast<fp16_t>(val);
                std::memcpy(ptr,&h,sizeof(fp16_t));break;
            }
            case LLAISYS_DTYPE_BF16:{
                bf16_t b=utils::cast<bf16_t>(val);
                std::memcpy(ptr,&b,sizeof(bf16_t));break;
            }
            default:EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    };
    for(size_t i = 0; i < in->shape()[0]; ++i) {
        for (size_t j = 0; j < weight->shape()[0]; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < in->shape()[1]; ++k)
                sum += get_float_at(in->data(), i * in->shape()[1] + k, in->dtype()) * get_float_at(weight->data(), j * in->shape()[1] + k, weight->dtype());
            if (bias)
                sum += get_float_at(bias->data(), j, bias->dtype());
            set_float_at(out->data(), i * weight->shape()[0] + j, sum, out->dtype());
        }
    }
}
} // namespace llaisys::ops