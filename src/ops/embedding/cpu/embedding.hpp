#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(
    void* out,
    const void* index,
    const void* weight,
    size_t index_size,
    size_t embedding_dim,
    llaisysDataType_t data_type
);
}