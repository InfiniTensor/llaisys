#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight);
void parallelEmbedding(tensor_t out, tensor_t index, tensor_t weight_local, size_t vocab_start, size_t vocab_end);
}
