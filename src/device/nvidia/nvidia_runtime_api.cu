#include "../runtime_api.hpp"
#include "cuda_utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device_id) {
    cuda_check(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    cuda_check(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream{};
    cuda_check(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    cuda_check(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}
void streamSynchronize(llaisysStream_t stream) {
    cuda_check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    cuda_check(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    cuda_check(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    cuda_check(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpyKind cuda_kind = cudaMemcpyDefault;
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        cuda_kind = cudaMemcpyHostToHost;
        break;
    case LLAISYS_MEMCPY_H2D:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
    case LLAISYS_MEMCPY_D2H:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
    case LLAISYS_MEMCPY_D2D:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
    default:
        cuda_kind = cudaMemcpyDefault;
        break;
    }
    cuda_check(cudaMemcpy(dst, src, size, cuda_kind));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaMemcpyKind cuda_kind = cudaMemcpyDefault;
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        cuda_kind = cudaMemcpyHostToHost;
        break;
    case LLAISYS_MEMCPY_H2D:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
    case LLAISYS_MEMCPY_D2H:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
    case LLAISYS_MEMCPY_D2D:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
    default:
        cuda_kind = cudaMemcpyDefault;
        break;
    }
    cuda_check(cudaMemcpyAsync(dst, src, size, cuda_kind, reinterpret_cast<cudaStream_t>(stream)));
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
