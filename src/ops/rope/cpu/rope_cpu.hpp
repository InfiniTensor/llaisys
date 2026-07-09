#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
// T *out (输出指针):意义: 指向输出张量数据的起始内存地址。
// 作用: 计算完成后，旋转后的 Query 或 Key 向量将按顺序存放在这里。

// const T *in (输入指针):意义: 指向原始输入张量（Q或K）数据的起始内存地址，布局通常为[seqlen, num_heads, d]。

// const int64_t *pos_ids (位置 ID 数组):
// 意义: 一个长度为 seqlen 的整数数组。作用: 存储了序列中每个 token 的绝对位置索引（Position Index）。
// RoPE 的旋转角度 $\phi$ 是位置的函数，因此 pos_ids[i] 直接决定了第 i 个 token 的旋转幅度
//  pos_ids[i] = i

// size_t seqlen (序列长度):
// 意义: 输入张量第一维的大小，即当前处理的 token 总数。

// size_t num_heads (注意力头数):
// 意义: 每个 token 拥有的注意力头数量。
// 作用: 用于计算内存偏移量。RoPE 对每个头执行相同的旋转逻辑（基于相同的位置 ID）。

// size_t d (头维度):
// 意义: 每个注意力头的特征维度（Head Dimension）。
// 约束: 必须为偶数。RoPE 将这 $d$ 维空间看作 $d/2$ 个二维平面，每个平面内的两个分量进行旋转。

// float theta (底数/基频):
// 意义: 用于计算频率因子的常数（通常为 10000.0）。
// 作用: 决定了不同维度上旋转速度的衰减率。

void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t dtype, size_t seqlen, size_t num_heads, size_t d, float theta);

} // namespace llaisys::ops::cpu