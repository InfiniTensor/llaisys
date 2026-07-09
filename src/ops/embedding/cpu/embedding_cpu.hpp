#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t dtype, size_t num_indices, size_t weight_rows, size_t weight_cols);
}