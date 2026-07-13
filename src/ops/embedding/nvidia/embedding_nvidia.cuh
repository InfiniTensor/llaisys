#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t num_indices,
               size_t hidden_size, llaisysStream_t stream);
void parallelEmbedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t num_indices,
                       size_t hidden_size, size_t vocab_start, size_t vocab_end, llaisysStream_t stream);
}
