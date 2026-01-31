#pragma once
#include "llaisys/tensor.h"

#include "../tensor/tensor.hpp"

__C {
    typedef struct LlaisysTensor {
        llaisys::tensor_t tensor;
    } LlaisysTensor;
}
