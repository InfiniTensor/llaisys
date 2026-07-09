#pragma once
#include "../../../core/llaisys_core.hpp"

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t rows, size_t cols, float eps);
}
