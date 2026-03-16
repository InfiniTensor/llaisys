/**
 * NVIDIA CUDA Runtime API 实现：设备管理、显存分配、同步与拷贝。
 * 对应 include/llaisys/runtime.h 中的 LlaisysRuntimeAPI。
 */
#include "../runtime_api.hpp"
#include "../../../include/llaisys.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace llaisys::device::nvidia {

namespace runtime_api {

static cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return cudaMemcpyDeviceToDevice;
    default: return cudaMemcpyHostToHost;
    }
}

int getDeviceCount() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        std::cerr << "[llaisys/nvidia] cudaGetDeviceCount failed: " << cudaGetErrorString(e)
                  << " (" << e << "). Check CUDA_VISIBLE_DEVICES and driver." << std::endl;
        return 0;
    }
    return n;
}

void setDevice(int id) {
    cudaSetDevice(id);
}

void deviceSynchronize() {
    cudaDeviceSynchronize();
}

llaisysStream_t createStream() {
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    return (llaisysStream_t)s;
}

void destroyStream(llaisysStream_t stream) {
    if (stream)
        cudaStreamDestroy((cudaStream_t)stream);
}

void streamSynchronize(llaisysStream_t stream) {
    if (stream)
        cudaStreamSynchronize((cudaStream_t)stream);
    else
        cudaDeviceSynchronize();
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr)
        cudaFree(ptr);
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    cudaMallocHost(&ptr, size);
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr)
        cudaFreeHost(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaStream_t s = stream ? (cudaStream_t)stream : (cudaStream_t)0;
    cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), s);
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
    &memcpyAsync,
};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::nvidia
