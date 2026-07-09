#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
int64_t sample(tensor_t logits, float temperature, int top_k, float top_p, uint64_t seed);
}
