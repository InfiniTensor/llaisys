#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::iluvatar {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t type, size_t num_rows, size_t row_size);
}
