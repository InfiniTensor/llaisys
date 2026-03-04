#pragma once

#include "../../../tensor/tensor.hpp"
#include <cstddef> // for size_t

namespace llaisys::ops::nvidia {

void rope(
    std::byte *out, 
    const std::byte *in, 
    const int64_t *pos_ids,
    llaisysDataType_t dtype,
    size_t seq_len, 
    size_t n_head, 
    size_t head_dim, 
    float theta
);

} // namespace llaisys::ops::nvidia