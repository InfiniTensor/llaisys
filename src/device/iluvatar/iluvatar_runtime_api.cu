#include "../runtime_api.hpp"

#include <cuda_runtime.h>

namespace llaisys::device::iluvatar {

namespace runtime_api {

// Helper function to convert llaisysMemcpyKind_t to cudaMemcpyKind
cudaMemcpyKind convertMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
        case LLAISYS_MEMCPY_H2H:
            return cudaMemcpyHostToHost;
        case LLAISYS_MEMCPY_H2D:
            return cudaMemcpyHostToDevice;
        case LLAISYS_MEMCPY_D2H:
            return cudaMemcpyDeviceToHost;
        case LLAISYS_MEMCPY_D2D:
            return cudaMemcpyDeviceToDevice;
        default:
            return cudaMemcpyHostToHost;
    }
}

int getDeviceCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

void setDevice(int device_id) {
    cudaSetDevice(device_id);
}

void deviceSynchronize() {
    cudaDeviceSynchronize();
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    cudaStreamDestroy((cudaStream_t)stream);
}

void streamSynchronize(llaisysStream_t stream) {
    cudaStreamSynchronize((cudaStream_t)stream);
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void freeDevice(void *ptr) {
    cudaFree(ptr);
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    cudaMallocHost(&ptr, size);
    return ptr;
}

void freeHost(void *ptr) {
    cudaFreeHost(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpy(dst, src, size, convertMemcpyKind(kind));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaMemcpyAsync(dst, src, size, convertMemcpyKind(kind), (cudaStream_t)stream);
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
} // namespace llaisys::device::iluvatar
