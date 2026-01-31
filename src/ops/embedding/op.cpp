#include "op.hpp"
#include <cstring>
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    size_t batch_size=index->numel();
    if(batch_size==0) return;
    size_t vocab_size=weight->shape()[0];
    size_t embed_dim=weight->shape()[1];
    const std::byte* weight_data=weight->data();
    std::byte* out_data=out->data();
    const int64_t* index_data=reinterpret_cast<const int64_t*>(index->data());
    llaisysDataType_t dtype=weight->dtype();
    size_t elem_size=utils::dsize(dtype);
    size_t row_size=embed_dim*elem_size;
    for(size_t i=0;i<batch_size;i++){
        int64_t idx=index_data[i];
        if(idx<0||idx>=static_cast<int64_t>(vocab_size))
            throw std::out_of_range("Index out of range in embedding");
        const std::byte* src_row=weight_data+idx*row_size;
        memcpy(out_data+i*row_size,src_row,row_size);
    }
}
} // namespace llaisys::ops
