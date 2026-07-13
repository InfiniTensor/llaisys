#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {

/**
 * MatMul: input=[N, M] * weight=[K, M], bias=None or [N, 1] -> output=[N, K]
 */
void linear(std::byte *output,
            const std::byte *input,
            const std::byte *weight,
            const std::byte *bias,
            size_t N,
            size_t M,
            size_t K,
            llaisysDataType_t dtype);

}