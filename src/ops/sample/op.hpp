#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

void sample(
    tensor_t sampled_token,
    tensor_t logits,
    float temperature,
    int top_k,
    float top_p
);

} // namespace llaisys::ops
