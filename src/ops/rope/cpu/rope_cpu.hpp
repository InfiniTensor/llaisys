#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

void rope(std::byte *output,
          const std::byte *input,
          const std::byte *pos_ids,
          size_t seqlen,
          size_t num_head,
          size_t head_dim,
          float theta,
          llaisysDataType_t dtype);

}