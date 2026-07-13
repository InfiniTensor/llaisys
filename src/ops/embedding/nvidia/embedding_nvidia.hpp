#pragma once

#include "../../../tensor/tensor.hpp"
#include <cstddef>

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t type, size_t num_indices, size_t vocab_size, size_t embedding_dim);
} // namespace llaisys::ops::nvidia