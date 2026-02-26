#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, float eps, size_t batch_size, size_t hidden_dim);
}
