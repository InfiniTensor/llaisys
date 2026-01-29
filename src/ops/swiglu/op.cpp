#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
template<typename T>
void swiglu_kernel(tensor_t out, tensor_t gate, tensor_t up){
    T*out_ptr=reinterpret_cast<T*>(out->data());
    const T*gate_ptr=reinterpret_cast<T*>(gate->data());
    const T*up_ptr=reinterpret_cast<T*>(up->data());
    size_t seqlen=out->shape()[0];
    size_t intermediate_size=out->shape()[1];
    for(size_t i=0;i<seqlen;i++){
        for(size_t j=0;j<intermediate_size;j++){
            float up_val=utils::cast<float>(up_ptr[i*intermediate_size+j]);
            float gate_val=utils::cast<float>(gate_ptr[i*intermediate_size+j]);
            float t_val=gate_val/(1+std::exp(-gate_val));
            float out_val=up_val*t_val;
            out_ptr[i*intermediate_size+j]=utils::cast<T>(out_val);
        }
    }
}
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    switch (out->dtype()) {
        case LLAISYS_DTYPE_F16:
            swiglu_kernel<llaisys::fp16_t>(out,gate,up);
            break;
        case LLAISYS_DTYPE_BF16:
            swiglu_kernel<llaisys::bf16_t>(out,gate,up);
            break;
        case LLAISYS_DTYPE_F32:
            swiglu_kernel<float>(out,gate,up);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops
