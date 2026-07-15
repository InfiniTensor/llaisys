#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
int64_t sample(tensor_t logits, int top_k, float top_p, float temperature);
}
