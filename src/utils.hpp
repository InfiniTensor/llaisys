#pragma once
#include "utils/check.hpp"
#include "utils/types.hpp"

#if defined(ENABLE_NVIDIA_API) && defined(__CUDACC__)
#include "utils/cuda_type_utils.cuh"
#endif
