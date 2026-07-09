#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, llaisysDataType_t type, size_t input_rows, size_t input_cols, size_t weight_rows);
} // namespace llaisys::ops::cpu
