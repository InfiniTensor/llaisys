#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {

void embedding(const std::byte *weight_data,
               const std::byte *index_data,
               std::byte *out_data,
               llaisysDataType_t dtype,
               size_t embedding_dim,
               size_t num_embeddings,
               size_t num_indices);

}