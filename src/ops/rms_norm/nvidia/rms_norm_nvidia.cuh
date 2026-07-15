#pragma once

#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              float eps,
              llaisysDataType_t dtype,
              const std::vector<size_t> &shape);
}
