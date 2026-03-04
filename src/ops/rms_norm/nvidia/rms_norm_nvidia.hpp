#pragma once

#include "../../../tensor/tensor.hpp"
#include <cstddef> // for size_t

namespace llaisys::ops::nvidia {

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t rows, size_t cols, float eps);

} // namespace llaisys::ops::nvidia