#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
// std::byte *attn_val (输出)
// 含义：指向存放计算结果的内存首地址。形状：[seqlen, nhead, dv]。
// 作用：存储最终的注意力输出 O = Softmax(QK^T)V。

// const std::byte *q (输入 Query)
// 含义：指向查询张量数据的内存首地址。形状：[seqlen, nhead, d]。
// 作用：代表当前需要查询的 Token 向量。

// const std::byte *k (输入 Key)
// 含义：指向键张量数据的内存首地址。形状：[total_len, nkvhead, d]。
// 作用：包含当前序列以及（可能的）KV Cache 中的历史 Key 向量。

// const std::byte *v (输入 Value)
// 含义：指向值张量数据的内存首地址。形状：[total_len, nkvhead, dv]。
// 作用：包含当前序列以及（可能的）KV Cache 中的历史 Value 向量

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t seqlen, size_t total_len, 
                    size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale);

//   元数据参数：                  
// llaisysDataType_t dtype：
// 含义：数据类型枚举（如 LLAISYS_DTYPE_F32, LLAISYS_DTYPE_F16）。
// 作用：决定了如何解释上述 std::byte* 指针。例如，如果是 F32，则每 4 个字节解析为一个数值。

// size_t seqlen
// 含义：当前处理的 Query 序列长度。

// size_t total_len
// 含义：Key 和 Value 的总长度。
// 公式：total_len = past_len (KV Cache 长度) + seqlen。
// 作用：用于确定 K 和 V 张量的第一维大小，以及计算因果掩码（Causal Mask）的边界。

// size_t nhead
// 含义：Query 的注意力头数量。

// size_t nkvhead
// 含义：Key 和 Value 的注意力头数量。
// GQA/MQA 支持：如果 nhead == nkvhead：标准多头注意力 (MHA)。
// 如果 nhead > nkvhead：分组查询注意力 (GQA)。此时 group_size = nhead / nkvhead，多个 Q 头共享一个 KV 头。

// size_t d
// 含义：Query 和 Key 的特征维度（Head Dimension）。
// 作用：决定了点积计算 $Q \cdot K^T$ 的向量长度。

// size_t dv
// 含义：Value 和输出 attn_val 的特征维度。
// 注意：虽然在 Llama 等模型中通常 d == dv，但从 API 设计角度允许它们不同。

// float scale
// 含义：缩放因子，通常取值为 $1 / \sqrt{d}$。
// 作用：在 Softmax 之前对分数进行缩放，防止数值过大导致梯度消失
} // namespace llaisys::ops::cpu