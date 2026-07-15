#pragma once

#include "../../core/llaisys_core.hpp"
#include "../../device/nvidia/cuda_utils.cuh"
#include "../../utils.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace llaisys::ops::nvidia {

inline cudaStream_t current_stream() {
    return reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
}

inline int current_device_id() {
    return core::context().runtime().deviceId();
}

class CublasHandlePool {
public:
    ~CublasHandlePool() {
        for (auto &entry : handles) {
            if (entry.second != nullptr) {
                cublasDestroy(entry.second);
            }
        }
    }

    std::unordered_map<int, cublasHandle_t> handles;
};

inline cublasHandle_t current_cublas_handle() {
    thread_local CublasHandlePool pool;
    cublasHandle_t &handle = pool.handles[current_device_id()];
    if (handle == nullptr) {
        LLAISYS_CUBLAS_CHECK(cublasCreate(&handle));
    }
    LLAISYS_CUBLAS_CHECK(cublasSetStream(handle, current_stream()));
    return handle;
}

struct TensorDescriptor {
    int ndim = 0;
    int64_t shape[8]{};
    int64_t strides[8]{};
};

inline TensorDescriptor make_descriptor(const std::vector<size_t> &shape, const std::vector<size_t> &strides) {
    ASSERT(shape.size() == strides.size(), "Tensor descriptor shape/stride rank mismatch.");
    ASSERT(shape.size() <= 8, "Tensor descriptor only supports up to 8 dimensions.");

    TensorDescriptor desc{};
    desc.ndim = static_cast<int>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        desc.shape[i] = static_cast<int64_t>(shape[i]);
        desc.strides[i] = static_cast<int64_t>(strides[i]);
    }
    return desc;
}

inline size_t numel(const std::vector<size_t> &shape) {
    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }
    return total;
}

inline bool is_contiguous(const std::vector<size_t> &shape, const std::vector<size_t> &stride) {
    if (shape.size() != stride.size()) {
        return false;
    }

    size_t expected = 1;
    for (std::ptrdiff_t dim = static_cast<std::ptrdiff_t>(shape.size()) - 1; dim >= 0; --dim) {
        if (stride[static_cast<size_t>(dim)] != expected) {
            return false;
        }
        expected *= shape[static_cast<size_t>(dim)];
    }
    return true;
}

inline int blocks_for(size_t total, int threads = 256) {
    return static_cast<int>((total + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
}

} // namespace llaisys::ops::nvidia
