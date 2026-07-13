#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

void embedding(std::byte *output,
               const std::byte *indices,
               const std::byte *weights,
               size_t num_indices,
               size_t embedding_dim,
               llaisysDataType_t dtype);

} // namespace llaisys::ops::nvidia
