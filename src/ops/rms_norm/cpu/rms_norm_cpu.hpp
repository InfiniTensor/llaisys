#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const float eps,
              llaisysDataType_t type, const size_t B, const size_t K);
}