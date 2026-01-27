#include "op.hpp"
#include "../../utils.hpp"
namespace llaisys::ops {
template<typename T>
void linear_cpu_kernel(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias){
    const T*in_ptr=reinterpret_cast<T*>(in->data());
    const T*weight_ptr=reinterpret_cast<T*>(weight->data());
    const T*bias_ptr=nullptr;
    T*out_ptr=reinterpret_cast<T*>(out->data());
    if(bias&&bias->numel()>0) bias_ptr=reinterpret_cast<T*>(bias->data());
    size_t M=in->shape()[0];
    size_t K=in->shape().back();
    size_t N=weight->shape()[0];
    for(size_t i=0;i<M;i++){
        for(size_t j=0;j<N;j++){
            float sum=0.0f;
            for(size_t index=0;index<K;index++){
                float x_val=utils::cast<float>(in_ptr[index+i*K]);
                float y_val=utils::cast<float>(weight_ptr[index+j*K]);
                sum+=x_val*y_val;
            }
            if(bias_ptr) sum+=utils::cast<float>(bias_ptr[j]);
            out_ptr[i*N+j]=utils::cast<T>(sum);
        }
    }
}
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F16:
            linear_cpu_kernel<llaisys::fp16_t>(out,in,weight,bias);
            break;
        case LLAISYS_DTYPE_BF16:
            linear_cpu_kernel<llaisys::bf16_t>(out,in,weight,bias);
            break;
        case LLAISYS_DTYPE_F32:
            linear_cpu_kernel<float>(out,in,weight,bias);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops
