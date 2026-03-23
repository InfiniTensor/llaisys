#pragma once

#include "../../../utils.hpp"

#include <cstddef>

namespace llaisys::ops::nvidia {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nhead, size_t nkvh, size_t dim,
                    size_t dv, float scale);
}
