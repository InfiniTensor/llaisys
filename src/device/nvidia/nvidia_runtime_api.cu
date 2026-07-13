#include "../runtime_api.hpp"

#include <cstring>
#include "cuda_utils.cuh"

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    CUDA_CHECK(cudaStreamDestroy(toCudaStream(stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    if (stream == nullptr) {
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(toCudaStream(stream)));
}

void *mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    CUDA_CHECK(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    CUDA_CHECK(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    if (size == 0) {
        return;
    }
    if (kind == LLAISYS_MEMCPY_H2H) {
        std::memcpy(dst, src, size);
        return;
    }
    CUDA_CHECK(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    if (size == 0) {
        return;
    }
    if (kind == LLAISYS_MEMCPY_H2H) {
        std::memcpy(dst, src, size);
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), toCudaStream(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
