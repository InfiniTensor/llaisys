#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
    void embedding(std::byte *out, const std::byte *index, const std::byte *wight, llaisysDataType_t dtype, size_t numel, size_t embedding_dim);
}
