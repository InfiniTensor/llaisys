#include "op.hpp"
#include <cstring>
namespace llaisys::ops {
template<typename T>
void embedding_cpu_kernal(tensor_t out, tensor_t index, tensor_t weight){
    const T*weight_val=reinterpret_cast<T*>(weight->data());
    T*out_val=reinterpret_cast<T*>(out->data());
    const int64_t*index_val=reinterpret_cast<int64_t*>(index->data());
    size_t embedding_dim=weight->shape().back();
    size_t n=index->numel();
    for(size_t i=0;i<n;i++){
        auto idx=index_val[i];
        const auto src_row=weight_val+idx*embedding_dim;
        auto dst_row=out_val+i*embedding_dim;
        std::memcpy(dst_row, src_row, sizeof(T)*embedding_dim);
    }
}
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    switch (weight->dtype()) {
        case LLAISYS_DTYPE_F16:
            embedding_cpu_kernal<llaisys::fp16_t>(out,index,weight);
            break;
        case LLAISYS_DTYPE_BF16:
            embedding_cpu_kernal<llaisys::bf16_t>(out,index,weight);
            break;
        case LLAISYS_DTYPE_F32:
            embedding_cpu_kernal<float>(out,index,weight);
            break;
        default:
            throw std::runtime_error("Not support this dtype!");
    }
}
} // namespace llaisys::ops
