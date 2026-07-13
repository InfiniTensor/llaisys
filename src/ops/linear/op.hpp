#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
void columnParallelLinear(tensor_t out_local, tensor_t in, tensor_t weight_local, tensor_t bias_local);
void rowParallelLinear(tensor_t out, tensor_t in_local, tensor_t weight_local);
tensor_t gatherLastDim(const tensor_t &local);
}
