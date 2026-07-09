#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void sample(tensor_t out_idx, tensor_t logits, float temperature, int top_k, float top_p);
}
