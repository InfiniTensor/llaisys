#pragma once

#include "../../tensor/tensor.hpp"

// 功能：按照索引(1-D)从权重矩阵(2-D)中抽取指定行，生成输出张量(2-D)，即将索引映射为稠密向量
// weight: 2-D tensor, shape: [num_embeddings, embedding_dim]
// index: 1-D tensor, shape: [batch_size]
// out: 2-D tensor, shape: [batch_size, embedding_dim]
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight);
}
