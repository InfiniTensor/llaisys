#pragma once

#include "../../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops {
void random_sample(
    tensor_t out_idx,
    tensor_t logits,
    float temperature,
    size_t top_k,
    float top_p,
    uint64_t seed);
}
