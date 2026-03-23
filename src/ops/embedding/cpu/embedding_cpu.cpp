#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type,
               size_t index_numel, size_t embd_dim, size_t weight_rows) {
    size_t elem_size = 0;
    switch (type) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        elem_size = llaisys::utils::dsize(type);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);
    size_t row_bytes = embd_dim * elem_size;

    for (size_t i = 0; i < index_numel; ++i) {
        int64_t idx = idx_ptr[i];
        ASSERT(idx >= 0 && static_cast<size_t>(idx) < weight_rows, "Embedding: index out of range.");
        const std::byte *src = weight + static_cast<size_t>(idx) * row_bytes;
        std::byte *dst = out + i * row_bytes;
        std::memcpy(dst, src, row_bytes);
    }
}
} // namespace llaisys::ops::cpu
