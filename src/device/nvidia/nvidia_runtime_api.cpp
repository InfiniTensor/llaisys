#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

namespace llaisys::device::nvidia {

namespace runtime_api {
static inline void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA runtime API failed");
    }
}

static inline cudaMemcpyKind to_cuda_memcpy_kind(llaisysMemcpyKind_t kind) {
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
        throw std::invalid_argument("Unsupported memcpy kind");
    }
}

int getDeviceCount() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) {
        return 0;
    }
    check_cuda(err, "cudaGetDeviceCount");
    return count;
}

void setDevice(int device_id) {
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
}

void deviceSynchronize() {
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    check_cuda(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamDestroy");
}

void streamSynchronize(llaisysStream_t stream) {
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, size), "cudaMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    check_cuda(cudaFree(ptr), "cudaFree");
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    check_cuda(cudaMallocHost(&ptr, size), "cudaMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    check_cuda(cudaFreeHost(ptr), "cudaFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    check_cuda(cudaMemcpy(dst, src, size, to_cuda_memcpy_kind(kind)), "cudaMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    check_cuda(
        cudaMemcpyAsync(dst, src, size, to_cuda_memcpy_kind(kind), reinterpret_cast<cudaStream_t>(stream)),
        "cudaMemcpyAsync");
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
