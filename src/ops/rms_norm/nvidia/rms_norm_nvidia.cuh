#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *input, const std::byte *weight, llaisysDataType_t type, size_t batch,
              size_t hidden_size, float eps, llaisysStream_t stream);
}
#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *input, const std::byte *weight, llaisysDataType_t type, size_t batch, size_t hidden_size, float eps);
}
