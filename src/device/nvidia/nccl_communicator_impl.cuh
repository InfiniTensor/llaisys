#pragma once

#include "nccl_communicator.hpp"
#include <nccl.h>
#include <cuda_runtime.h>

// NCCL error checking macro
#define NCCL_CHECK(cmd)                                                        \
    do {                                                                       \
        ncclResult_t result = cmd;                                             \
        if (result != ncclSuccess) {                                           \
            throw std::runtime_error(std::string("NCCL error: ") +             \
                                     ncclGetErrorString(result));              \
        }                                                                      \
    } while (0)

// CUDA error checking macro
#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        cudaError_t result = cmd;                                              \
        if (result != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(result));              \
        }                                                                      \
    } while (0)
