#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t type, size_t num_indices, size_t vocab_size, size_t embedding_dim);
}