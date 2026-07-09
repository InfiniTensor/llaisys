#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
// GPU embedding lookup：每个 block 处理一行，通过合并访存提升带宽效率
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t num_indices, size_t embedding_dim);
} // namespace llaisys::ops::nvidia
