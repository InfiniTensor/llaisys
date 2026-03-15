#pragma once

#include "../../tensor/tensor.hpp"

// 功能：计算线性变换，即matmul
// in/X: 形状[M, K]
// weight/W: 形状[N, K]，存的是未转置的W
// bias/b: 形状[N]（可选；为 nullptr 时等价于不加 bias）
// out/Y: 形状[M, N]
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias = nullptr);
}
