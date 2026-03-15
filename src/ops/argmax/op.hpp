#pragma once

#include "../../tensor/tensor.hpp"

// C++对外(python)暴露的接口声明
// 功能：获取张量vals的最大值及其索引，并分别存储在max_val和max_idx中
namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}