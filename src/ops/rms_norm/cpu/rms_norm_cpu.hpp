#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *input, const std::byte *weight, llaisysDataType_t type, size_t batch, size_t hidden_size, float eps);
}
