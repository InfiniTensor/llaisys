#include "embedding_cpu.hpp"
#include "../../../utils.hpp"

#include <cstring> // std::memcpy

namespace llaisys::ops::cpu {

void embedding(std::byte *out,
               const int64_t *index,
               const std::byte *weight,
               llaisysDataType_t type,
               size_t N, size_t D, size_t V) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const size_t elem = llaisys::utils::dsize(type);
    const size_t row_bytes = D * elem;

    for (size_t i = 0; i < N; ++i) {
        const int64_t idx = index[i];
        ASSERT(idx >= 0 && static_cast<size_t>(idx) < V, "embedding: index out of range");

        std::byte *dst = out + i * row_bytes;
        const std::byte *src = weight + static_cast<size_t>(idx) * row_bytes;
        std::memcpy(dst, src, row_bytes);
    }
}

} // namespace llaisys::ops::cpu