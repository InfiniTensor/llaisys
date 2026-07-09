#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void rand_sample(tensor_t sample_idx, tensor_t sample_val, tensor_t vals, float temperature, size_t topK, float topP,
                 int64_t seed);
}
