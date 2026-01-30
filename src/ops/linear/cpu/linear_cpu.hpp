#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
// 指针类参数（内存地址）
// std::byte *out (输出 Y)
// 意义：指向计算结果存放的内存首地址。
// 形状：逻辑形状为 $[M, N]$。
// 作用：计算完成后，这里将存储经过线性变换后的特征数据。

// const std::byte *in (输入 X)
// 意义：指向输入特征矩阵的内存首地址。
// 形状：逻辑形状为 $[M, K]$。
// 作用：作为矩阵乘法的左矩阵。

// const std::byte *weight (权重 W)
// 意义：指向权重矩阵的内存首地址。
// 形状：逻辑形状为 $[N, K]$。
// 注意：公式中是 $W^T$（转置），但在内存中 $W$ 是按 $[N, K]$ 连续存储的。代码通过逻辑索引 weight[j * K + k] 实现了在不物理转置内存的情况下进行转置乘法。

// const std::byte *bias (偏置 b)
// 意义：指向偏置向量的内存首地址。
// 形状：逻辑形状为 $[N]$。
// 可选性：如果不需要偏置，该指针可以为 nullptr。
void linear(std::byte *out, 
            const std::byte *in, 
            const std::byte *weight, 
            const std::byte *bias,
            llaisysDataType_t dtype, size_t M, size_t N, size_t K);
// 元数据与维度参数：
// llaisysDataType_t dtype
// 意义：数据类型枚举（如 float32, float16, bfloat16）。
// 作用：决定了内核函数如何解释 std::byte 指针。例如，如果是 float32，则每 4 个字节解析为一个数值。

// size_t M
// 意义：输入矩阵 $X$ 的行数。
// 深度学习背景：通常代表 Batch Size（批大小）或者序列长度。

// size_t N
// 意义：输出特征的维度（Output Features）。
// 对应关系：它等于权重矩阵 $W$ 的行数，也是偏置 $b$ 的长度，以及输出 $Y$ 的列数。

// size_t K
// 意义：输入特征的维度（Input Features）。
// 对应关系：它等于输入 $X$ 的列数，同时也必须等于权重矩阵 $W$ 的列数（矩阵乘法的内维必须匹配）。
} // namespace llaisys::ops::cpu