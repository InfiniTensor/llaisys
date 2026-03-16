/**
 * Add 算子对外接口声明。
 *
 * 语义：逐元素加法 c = a + b。c、a、b 为同形状、同 dtype、同设备的张量，
 * 调用方需保证 c 已分配好内存；本算子只负责写入 c，不负责分配。
 */
#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

/// 逐元素加法：c[i] = a[i] + b[i]，c / a / b 需同 shape、同 dtype、同 device，且连续
void add(tensor_t c, tensor_t a, tensor_t b);
}
