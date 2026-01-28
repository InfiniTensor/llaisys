#include "op.hpp"
#include <cmath>
const int INF=0x3f3f3f3f;
namespace llaisys::ops {
template<typename T>
void self_attention_kernel(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale){
    T*attn_val_ptr=reinterpret_cast<T*>(attn_val->data());
    const T*q_ptr=reinterpret_cast<T*>(q->data());
    const T*k_ptr=reinterpret_cast<T*>(k->data());
    const T*v_ptr=reinterpret_cast<T*>(v->data());
    size_t seqlen=q->shape()[0],nhead=q->shape()[1],d=q->shape()[2];
    size_t total_len=k->shape()[0],nkvhead=k->shape()[1];
    size_t dv=v->shape()[2];
    std::vector<T> A(total_len);
    auto get_k_index=[&](size_t n_index)->size_t{
        if(nkvhead==1) return 0;
        if(nkvhead==nhead) return n_index;
        return n_index/(nhead/nkvhead); 
    };
    for(size_t n_index=0;n_index<nhead;n_index++){
        size_t k_index=get_k_index(n_index);
        for(size_t i=0;i<seqlen;i++){
            float MAX_num=-1e30f;//一行的最大指数,后续归一化
            for(size_t j=0;j<total_len;j++){
                float sum=0.0f;
                for(size_t k=0;k<d;k++){
                    float q_val=utils::cast<float>(q_ptr[i*(nhead*d)+n_index*d+k]);
                    float k_val=utils::cast<float>(k_ptr[j*(nkvhead*d)+k_index*d+k]);
                    sum+=q_val*k_val;
                }
                sum*=scale;
                size_t global_i=total_len-seqlen+i;
                if(j>global_i){
                    A[j]=utils::cast<T>(-INF);
                }
                else{
                    A[j]=utils::cast<T>(sum);
                    MAX_num=fmax(MAX_num,sum);
                }
            }
            float softmax_accu=0;
            for(size_t j=0;j<total_len;j++){
                float a_val=utils::cast<float>(A[j]);
                softmax_accu+=std::exp(a_val-MAX_num);
            }
            float inv_accu=1.0f/softmax_accu;
            for(size_t j=0;j<total_len;j++){
                float a_val=utils::cast<float>(A[j]);
                float a_sval=std::exp(a_val-MAX_num);
                A[j]=utils::cast<T>(a_sval*inv_accu);
            }
            for(size_t j=0;j<dv;j++){
                float out_sum=0.0f;
                for(size_t k=0;k<total_len;k++){
                    float a_cv=utils::cast<float>(A[k]);
                    float v_cv=utils::cast<float>(v_ptr[k*nkvhead*dv+k_index*dv+j]);
                    out_sum+=a_cv*v_cv;
                }
                attn_val_ptr[i*nhead*dv+n_index*dv+j]=utils::cast<T>(out_sum);
            }
        }
    }
}
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    switch (q->dtype()) {
        case LLAISYS_DTYPE_F16:
            self_attention_kernel<llaisys::fp16_t>(attn_val,q,k,v,scale);
            break;
        case LLAISYS_DTYPE_BF16:
            self_attention_kernel<llaisys::bf16_t>(attn_val,q,k,v,scale);
            break;
        case LLAISYS_DTYPE_F32:
            self_attention_kernel<float>(attn_val,q,k,v,scale);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops
