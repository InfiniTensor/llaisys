#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

// 1.const std::byte *weight (权重矩阵)
//   意义：这是 Embedding 层存储的“词表”或“查找表”。
//   形状：逻辑上是一个 2D 矩阵，形状为 [num_embeddings, embedding_dim]。
//   作用：函数会根据 index 提供的值，找到 weight 中对应的行并将其复制出来。使用 std::byte 是为了处理通用内存，实际指向的数据类型由 val_dtype 决定。
// 2. const std::byte *index (索引张量)
//   意义：包含了你想要从 weight 中提取的行的索引。
//   形状：通常是一个 1D 张量，长度为 seq_len。
//   要求：根据你的项目需求，这里的索引必须是 Int64 类型（即 long long）。每个索引值 i 必须满足 0 <= i < num_embeddings。
// 3. std::byte *out (输出缓冲区)
//   意义：这是存放提取结果的目标内存区域。
//   形状：形状为 [seq_len, embedding_dim]。
//   作用：函数执行完成后，out 将包含 seq_len 个词向量，每个向量的长度为 embedding_dim。
// 4. llaisysDataType_t val_dtype (数据类型)
//   意义：指明 weight 和 out 中存储的数值是什么类型。
//   常见值：LLAISYS_DTYPE_F32 (float), LLAISYS_DTYPE_F16 (fp16), LLAISYS_DTYPE_BF16 (bf16)。
//   作用：内核函数（Kernel）需要根据这个参数来决定每个元素占用多少字节（例如 float 占 4 字节），从而正确计算内存偏移量并进行拷贝。
// 5. size_t seq_len (序列长度)
//   意义：本次操作要提取的向量个数。
//   来源：它等于 index 张量中元素的总数。
//   作用：决定了函数最外层循环的次数。
// 6. size_t embedding_dim (词向量维度)
//   意义：每个词向量包含的特征数量（即 weight 矩阵的列数）。
//   作用：决定了每一行数据的大小。在拷贝时，单次 memcpy 的字节数等于 embedding_dim * sizeof(val_dtype)。
void embedding(
    const std::byte *weight,       // 1. 权重矩阵指针
    const std::byte *index,        // 2. 索引张量指针
    std::byte *out,                // 3. 输出缓冲区指针
    llaisysDataType_t val_dtype,   // 4. 数据类型枚举
    size_t seq_len,                // 5. 序列长度（索引数量）
    size_t embedding_dim           // 6. 词向量维度
); // namespace llaisys::ops::cpu
}