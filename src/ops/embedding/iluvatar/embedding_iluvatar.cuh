#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::iluvatar {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t type, size_t index_size, size_t embd_dim);
}
