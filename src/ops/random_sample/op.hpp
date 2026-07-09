#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void random_sample(tensor_t sample_idx, tensor_t sample_val, tensor_t logits, float temperature, int top_k, float top_p,
				   int64_t seed);
}
