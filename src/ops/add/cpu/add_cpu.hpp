/**
 * Add 算子的 CPU 实现声明。
 *
 * 接口使用裸指针 + dtype + 元素个数，由 op.cpp 在通过张量合法性检查后调用；
 * 不关心张量形状布局，假定内存连续、按 numel 逐元素计算即可。
 */
#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

/// 逐元素加法 c[i]=a[i]+b[i]，c/a/b 为同一 dtype 的连续内存，size 为元素个数
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t size);
}
