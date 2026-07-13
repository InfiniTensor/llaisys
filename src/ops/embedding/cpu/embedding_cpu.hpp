#pragma once
#include "llaisys.h"
#include<cstddef>

namespace llaisys::ops::cpu {
    void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype, size_t num_indices, size_t embedding_dim);
} // namespace llaisys::ops::cpu