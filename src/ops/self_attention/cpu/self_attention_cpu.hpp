#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {

void self_attention(std::byte* out,
                    const std::byte* q,
                    const std::byte* k,
                    const std::byte* v,
                    llaisysDataType_t dtype,
                    size_t qlen,
                    size_t kvlen,
                    size_t nhead,
                    size_t nkvhead,
                    size_t head_dim,
                    size_t value_dim,
                    float scale);

}