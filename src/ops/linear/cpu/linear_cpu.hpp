#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, std::vector<size_t> out_shape, std::vector<size_t> in_shape, std::vector<size_t> weight_shape, std::vector<size_t> bias_shape);
}