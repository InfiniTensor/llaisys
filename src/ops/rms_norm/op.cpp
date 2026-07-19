#include "op.hpp"
#include <cmath>
namespace llaisys::ops {
template<typename T>
void rms_norm_cpu_kernel(tensor_t out, tensor_t in, tensor_t weight, float eps){
    T*out_ptr=reinterpret_cast<T*>(out->data());
    const T*in_ptr=reinterpret_cast<T*>(in->data());
    const T*weight_ptr=reinterpret_cast<T*>(weight->data());
    size_t d=in->shape().back();
    size_t n = in->numel() / d;
    for(size_t i=0;i<n;i++){//这里对每一行去计算
        float sum=0.0f;
        for(size_t j=0;j<d;j++){
            float cur_num=utils::cast<float>(in_ptr[i*d+j]);
            sum+=cur_num*cur_num;
        }
        sum/=(float)d;
        float std_sum=sqrtf(sum+eps);
        for(size_t j=0;j<d;j++){
            float wi=utils::cast<float>(weight_ptr[j]);
            float xi=utils::cast<float>(in_ptr[i*d+j]);
            float out_val=wi*xi/std_sum;
            out_ptr[i*d+j]=utils::cast<T>(out_val);
        }
    }
}
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    tensor_t contiguous_in = in->isContiguous() ? in : in->contiguous();
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F16:
            rms_norm_cpu_kernel<llaisys::fp16_t>(out,contiguous_in,weight,eps);
            break;
        case LLAISYS_DTYPE_BF16:
            rms_norm_cpu_kernel<llaisys::bf16_t>(out,contiguous_in,weight,eps);
            break;
        case LLAISYS_DTYPE_F32:
            rms_norm_cpu_kernel<float>(out,contiguous_in,weight,eps);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops
