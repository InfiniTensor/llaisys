#pragma once
#include "llaisys/tensor.h"

#include "../tensor/tensor.hpp"

LLAISYS_EXTERN_C {
    typedef struct LlaisysTensor {
        llaisys::tensor_t tensor;
    } LlaisysTensor;
}
