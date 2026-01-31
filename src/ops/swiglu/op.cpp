#include "op.hpp"
#include <cmath>
#include <cstring>
namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
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
    for(size_t i=0;i<out->shape()[0];i++)
        for(size_t j=0;j<out->shape()[1];j++){
            size_t idx=i*out->shape()[1]+j;
            auto gate_val=get_float_at(gate->data(),idx,gate->dtype());
            auto up_val=get_float_at(up->data(),idx,up->dtype());
            auto out_val=up_val*gate_val/(1+expf(-gate_val));
            set_float_at(out->data(),idx,out_val,out->dtype());
        }
}
} // namespace llaisys::ops
