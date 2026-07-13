#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <omp.h>

template <typename T>
static void
embedding_impl(T *output, const int64_t *indices, const T *weights, size_t num_indices, size_t embedding_dim) {
#pragma omp parallel for
    for (size_t i = 0; i < num_indices; i++) {
        int64_t idx = indices[i];
        const T *weight_begin = weights + idx * embedding_dim;
        T *output_begin = output + i * embedding_dim;
        std::memcpy(output_begin, weight_begin, embedding_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {

void embedding(std::byte *output,
               const std::byte *indices,
               const std::byte *weights,
               size_t num_indices,
               size_t embedding_dim,
               llaisysDataType_t dtype) {
    using namespace llaisys::utils;

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_impl(recast(float *, output), recast(const int64_t *, indices), recast(const float *, weights),
                              num_indices, embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_impl(recast(fp16_t *, output), recast(const int64_t *, indices),
                              recast(const fp16_t *, weights), num_indices, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_impl(recast(bf16_t *, output), recast(const int64_t *, indices),
                              recast(const bf16_t *, weights), num_indices, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu