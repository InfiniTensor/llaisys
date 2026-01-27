#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
template<typename T>
void rope_cpu_kernel(tensor_t out, tensor_t in, tensor_t pos_ids, float theta){
    T*out_ptr=reinterpret_cast<T*>(out->data());
    T*in_ptr=reinterpret_cast<T*>(in->data());
    int64_t*pos_ptr=reinterpret_cast<int64_t*>(pos_ids->data());
    size_t N=in->shape()[0],M=in->shape()[1],D=in->shape()[2];
    for(size_t i=0;i<N;i++){
        for(size_t j=0;j<M;j++){
            for(size_t k=0;k<D/2;k++){
                float a_in=utils::cast<float>(in_ptr[i*(M*D)+j*D+k]),b_in=utils::cast<float>(in_ptr[i*(M*D)+j*D+k+D/2]);
                float p_i=utils::cast<float>(pos_ptr[i]);
                float theta_in=p_i/powf(theta, 2.0f*k/D);
                T a_out=utils::cast<T>(a_in*cos(theta_in)-b_in*sin(theta_in));
                T b_out=utils::cast<T>(b_in*cos(theta_in)+a_in*sin(theta_in));
                out_ptr[i*(M*D)+j*D+k]=a_out;out_ptr[i*(M*D)+j*D+k+D/2]=b_out;
            }
        }
    }
}
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F16:
            rope_cpu_kernel<llaisys::fp16_t>(out,in,pos_ids,theta);
            break;
        case LLAISYS_DTYPE_BF16:
            rope_cpu_kernel<llaisys::bf16_t>(out,in,pos_ids,theta);
            break;
        case LLAISYS_DTYPE_F32:
            rope_cpu_kernel<float>(out,in,pos_ids,theta);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops
