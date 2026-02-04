#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const size_t i_size,
               const std::byte *weight, const size_t w_rows, const size_t w_cols,
               llaisysDataType_t type);
}