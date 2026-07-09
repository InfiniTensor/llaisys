#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t n_idx, size_t embd_dim) {
    auto *idx = reinterpret_cast<const int64_t *>(index);
    size_t esize = llaisys::utils::dsize(dtype);
    size_t row_bytes = embd_dim * esize;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_idx; i++) {
        int64_t row = idx[i];
        std::memcpy(out + i * row_bytes, weight + row * row_bytes, row_bytes);
    }
}
} // namespace llaisys::ops::cpu
