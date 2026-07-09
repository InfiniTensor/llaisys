#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rand_sample(std::byte *sample_idx, std::byte *sample_val, const std::byte *vals, llaisysDataType_t type, size_t numel,
                 const int64_t batch_size, const float temperature, const size_t topK, const float topP, const int64_t seed);
}
