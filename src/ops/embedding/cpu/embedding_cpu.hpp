#pragma once
#include "llaisys.h"

#include <cstddef>

// CPU 侧 embedding 接口：
// out   : [seqlen, embedding_dim]
// index : [seqlen]，int64 索引（
// weight: [num_embeddings, embedding_dim]
// type  : out/weight 的数据类型（F32/F16/BF16）
// index_numel    : seqlen
// embedding_dim  : 每个 embedding 向量的维度
namespace llaisys::ops::cpu {
void embedding(std::byte *out,
               const std::byte *index,
               const std::byte *weight,
               llaisysDataType_t type,
               size_t index_numel,
               size_t embedding_dim);
}