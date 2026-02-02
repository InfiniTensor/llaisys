#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

// tensor_t attn_val (输出张量)
// 意义：这是计算结果的存储位置，即注意力机制的最终输出。
// 数学角色：公式 Attention(Q,K,V)=Softmax(QK^T/sqrt(d))V 中的最终结果。
// 形状 (Shape)：[seqlen, nhead, dv]。
// 它代表了每个查询 Token 经过注意力加权后得到的特征向量。

// tensor_t q (查询张量 - Query)
// 意义：代表当前的“问题”或“搜索项”。模型通过 Q 去寻找序列中哪些部分是重要的。
// 形状 (Shape)：[seqlen, nhead, d]。
// seqlen: 当前处理的 Token 数量。nhead: 查询头的数量。d: 每个头的维度

// tensor_t k (键张量 - Key)
// 意义：代表序列中信息的“索引”或“标签”。Q 会与 K 进行点积计算，以确定相关性分数。
// 形状 (Shape)：[total_len, nkvhead, d]。
// total_len: 键的总长度。在推理模式下，这通常包含之前的 KV Cache 长度。nkvhead: 键/值头的数量。
// 注意：代码支持 GQA,如果 nhead > nkvhead，多个 Q 头会共享同一个 K 头。

// tensor_t v (值张量 - Value)
// 意义：代表序列中实际的“信息内容”。一旦通过 Q 和 K 确定了权重（Softmax 结果），就对 V 进行加权求和。
// 形状 (Shape)：[total_len, nkvhead, dv]。dv: 值向量的维度（通常与 d 相同，但实现上允许不同）。

// float scale (缩放因子)
// 意义：用于缩放 Q 和 K 的点积结果。 数学角色：公式中的 1/sqrt(d)。
// 作用：防止点积结果过大。如果点积值太大，经过 Softmax 后梯度会变得非常小（导致梯度消失问题）。通常取值为1.0/sqrt(d)。
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
}
