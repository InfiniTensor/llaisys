#pragma once
#include "../../../core/llaisys_core.hpp"

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t hidden_size);
}
