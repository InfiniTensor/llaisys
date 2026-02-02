#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// tensor_t out (输出张量)
// 含义：用于存储计算结果的张量。
// 数学角色：对应公式中的 out = up .* SiLU(gate)
// 形状 (Shape)：必须与输入张量 gate 和 up 完全一致。通常为 [seqlen, intermediate_size]。
// 数据类型：必须与输入张量一致。

// 2. tensor_t gate (门控张量)
// 含义：作为“门控”信号的输入张量。
// 数学角色：对应公式中的 SiLU(X) = x * σ(x) = x * 1/(1+exp(-x))
// 它会先经过 SiLU  激活函数处理：激活后的结果用于控制（缩放）up 张量的信息流。
// 形状 (Shape)：[seqlen, intermediate_size]。

// 3. tensor_t up (上投影张量)
// 含义：承载主要信息的输入张量（通常是线性变换后的结果）。
// 数学角色：对应公式中的 up。它与激活后的 gate 进行逐元素乘法 。
// 形状 (Shape)：[seqlen, intermediate_size]。

void swiglu(tensor_t out, tensor_t gate, tensor_t up);
}
