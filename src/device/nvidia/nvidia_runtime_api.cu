#include "../runtime_api.hpp"
#include "cuda_utils.cuh"

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    LLAISYS_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device_id) {
    LLAISYS_CUDA_CHECK(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    LLAISYS_CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    LLAISYS_CUDA_CHECK(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    LLAISYS_CUDA_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}
void streamSynchronize(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    LLAISYS_CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    LLAISYS_CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    LLAISYS_CUDA_CHECK(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    LLAISYS_CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    LLAISYS_CUDA_CHECK(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    LLAISYS_CUDA_CHECK(cudaDeviceSynchronize());
    LLAISYS_CUDA_CHECK(cudaMemcpy(dst, src, size, cuda_utils::memcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(
        dst,
        src,
        size,
        cuda_utils::memcpyKind(kind),
        reinterpret_cast<cudaStream_t>(stream)));
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
