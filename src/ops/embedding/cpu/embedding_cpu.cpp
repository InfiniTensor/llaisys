#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstring>
#include <cstddef>

namespace llaisys::ops::cpu {

template <typename T>
void embedding_t(std::uint8_t* out_ptr,
                    const int64_t* index_ptr,
                    const std::uint8_t* weight_ptr,
                    size_t index_num,
                    size_t embed_dim) {
    for (size_t i = 0;i <index_num; ++i) {
        const int64_t row = index_ptr[i];
        const T* src = reinterpret_cast<const T*>(weight_ptr) + row * embed_dim;
        T* dst = reinterpret_cast<T*>(out_ptr) + i * embed_dim;
        std::memcpy(dst, src, embed_dim * sizeof(T));
    }
}
void embedding(std::uint8_t* out_ptr,
            const int64_t* index_ptr,
            const std::uint8_t* weight_ptr,
            llaisysDataType_t dtype,
            size_t index_num,
            size_t embed_dim,
            size_t vocab){
    for (size_t i = 0; i < index_num; ++i)
    {
        ASSERT(index_ptr[i] >= 0 && static_cast<size_t>(index_ptr[i]) < vocab,
                "Index value out of bounds");
    }

    switch (dtype)
    {
    case LLAISYS_DTYPE_F32:
        return embedding_t<float>(out_ptr, index_ptr, weight_ptr, index_num, embed_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_t<llaisys::fp16_t>(out_ptr, index_ptr, weight_ptr, index_num, embed_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_t<llaisys::bf16_t>(out_ptr, index_ptr, weight_ptr, index_num, embed_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    }
} // namespace llaisys::ops::cpu 