#include "op.hpp"

namespace llaisys::ops {
template<typename T>
void rearrange_kernel(tensor_t out, tensor_t in){
    const T*in_ptr=reinterpret_cast<T*>(in->data());
    T*out_ptr=reinterpret_cast<T*>(out->data());
    size_t total_num=in->numel();
    size_t dim=in->ndim();
    for(size_t i=0;i<total_num;i++){
        size_t index_acc=i;
        size_t in_offset=0;
        size_t out_offset=0;
        for (size_t j = dim; j-- > 0;) {
            size_t cur_shape=in->shape()[j];
            size_t cur_index=index_acc%cur_shape;
            index_acc/=cur_shape;
            in_offset+=cur_index*in->strides()[j];
            out_offset+=cur_index*out->strides()[j];
        }
        out_ptr[out_offset]=in_ptr[in_offset];
    }  
}
void rearrange(tensor_t out, tensor_t in) {
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F16:
            rearrange_kernel<llaisys::fp16_t>(out,in);
            break;
        case LLAISYS_DTYPE_BF16:
            rearrange_kernel<llaisys::bf16_t>(out,in);
            break;
        case LLAISYS_DTYPE_F32:
            rearrange_kernel<float>(out,in);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops
