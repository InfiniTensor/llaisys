#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
              llaisysDataType_t type, size_t rows, size_t dim, float eps);
}