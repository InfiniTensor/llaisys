#include "op.hpp"
#include "../../utils.hpp"
#include "./cpu/rope_cpu.hpp"


// tensor_t out (输出张量)
// 意义：用于存储旋转后的结果。
// 数学角色：公式中的 $y_i = [a'_i, b'_i]$。
// 要求：形状：必须与输入 in 完全一致，通常为 [seqlen, nhead, d]。

// tensor_t in (输入张量)
// 意义：原始的 Query (Q) 或 Key (K) 张量。
// 数学角色：公式中的 x_i = [a_i, b_i]。
// 形状：[seqlen, nhead, d]。seqlen：序列长度（Token 数量），nhead：注意力头的数量。
// d：每个头的维度（Head Dimension）。
// 约束：维度 d 必须是偶数（代码中通过 d % 2 == 0 校验），因为 RoPE 是将向量两两配对进行旋转的。

// tensor_t pos_ids (位置 ID 张量)
// 意义：输入序列中每个 Token 在整个上下文中的绝对位置索引。
// 数学角色：公式中的 p_i。形状：[seqlen]（1D 张量）。其长度必须等于输入张量的第一维。
// 数据类型：必须是 int64 (LLAISYS_DTYPE_I64)。作用：它决定了旋转的角度大小。位置越靠后，旋转的角度倍数越大。

// float theta (底数/基频)，意义：频率向量的基准值。
// 数学角色：公式中的 theta。作用：用于计算不同维度的旋转频率。
// 在 Llama 中，这个值通常默认为 10000.0。增大这个值（如长文本优化中的NTK-Aware缩放）可以改变模型对长距离位置的感知

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 1. 基础校验
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64");

    // 2. 形状校验
    CHECK_ARGUMENT(in->ndim() == 3, "RoPE: input must be 3D [seqlen, nhead, d]");
    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];// 头维度
    size_t d = in->shape()[2];// 每个token的维度

    CHECK_ARGUMENT(pos_ids->numel() == seqlen, "RoPE: pos_ids length mismatch");
    CHECK_ARGUMENT(d % 2 == 0, "RoPE: head_dim must be even");
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    // 3. 连续性校验
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), 
           "RoPE: tensors must be contiguous");

    // 4. 设备分发
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::rope(out->data(), in->data(), (const int64_t *)pos_ids->data(), 
                  in->dtype(), seqlen, nhead, d, theta);
        return;
    }

    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
