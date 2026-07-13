#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

void rms_norm(std::byte *output,
              const std::byte *input,
              const std::byte *weight,
              size_t N,
              size_t M,
              float eps,
              llaisysDataType_t dtype);

}