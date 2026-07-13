#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

void embedding(std::byte *output,
               const std::byte *indices,
               const std::byte *weights,
               size_t num_indices,
               size_t embedding_dim,
               llaisysDataType_t dtype);

}