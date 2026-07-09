#pragma once

#include "llaisys.h"

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, std::byte *gate, std::byte *up, llaisysDataType_t dtype, size_t seqlen, size_t intermediate_size);
}