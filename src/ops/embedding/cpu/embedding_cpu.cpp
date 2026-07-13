#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include<cstring>

namespace{
    template<typename T>
    void embedding_(T* out,const int64_t *index,const T *weight,size_t num_indices,size_t embedding_dim){
        size_t stride_bytes=embedding_dim *sizeof(T);

        for(size_t i=0;i<num_indices;++i)
        {
            int64_t row_id=index[i];
            const T* src_row=weight+row_id*embedding_dim;
            T* dst_row=out+i*embedding_dim;
            std::memcpy(dst_row,src_row,stride_bytes);
        }
    }
}

namespace llaisys::ops::cpu {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t dtype, size_t num_indices, size_t embedding_dim) {
    
    const int64_t *index_ptr = reinterpret_cast<const int64_t *>(index);

    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return embedding_(reinterpret_cast<float*>(out), index_ptr, 
                              reinterpret_cast<const float*>(weight), num_indices, embedding_dim);
        case LLAISYS_DTYPE_F16:
            return embedding_(reinterpret_cast<llaisys::fp16_t*>(out), index_ptr, 
                              reinterpret_cast<const llaisys::fp16_t*>(weight), num_indices, embedding_dim);
        case LLAISYS_DTYPE_BF16:
            return embedding_(reinterpret_cast<llaisys::bf16_t*>(out), index_ptr, 
                              reinterpret_cast<const llaisys::bf16_t*>(weight), num_indices, embedding_dim);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu