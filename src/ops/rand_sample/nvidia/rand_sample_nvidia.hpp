#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {
void rand_sample(std::byte *sample_idx, std::byte *sample_val, const std::byte *vals, llaisysDataType_t type, size_t numel,
                 int64_t batch_size, float temperature, size_t topK, float topP, int64_t seed);
}
