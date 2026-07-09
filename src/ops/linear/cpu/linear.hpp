#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(
    void* out,
    const void* in,
    const void* weight,
    const void* bias,
    size_t batch_size,
    size_t in_features,
    size_t out_features,
    llaisysDataType_t data_type
);
}