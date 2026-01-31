#pragma once
#include "../../../tensor/tensor.hpp"
#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::uint8_t* out,
               const int64_t* index,
               const std::uint8_t* weight,
               llaisysDataType_t dtype,
               size_t index_num,
               size_t embed_dim,
               size_t vocab);
} // namespace llaisys::ops::cpu 