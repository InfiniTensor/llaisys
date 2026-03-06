#include "../runtime_api.hpp"
#include <cuda_runtime.h>
#include <iostream>

// CUDA 错误检查宏：帮你快速定位显存分配或执行错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace llaisys::device::nvidia {

namespace runtime_api {

int getDeviceCount() {
    int count = 0;
    cudaGetDeviceCount(&count); // 如果没有GPU，我们不希望它崩溃，所以这里不用 CHECK
    return count;
}

void setDevice(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

void deviceSynchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream) {
        CUDA_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
    }
}

void streamSynchronize(llaisysStream_t stream) {
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
    }
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    // 使用 Pinned Memory (锁页内存)，这能让 CPU <-> GPU 的异步数据拷贝快得多
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    // 现代 64 位 Linux 默认支持 UVA，cudaMemcpyDefault 会根据指针地址自动判断拷贝方向
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
}

// 修复点：添加了 llaisysStream_t 参数，并调用 cudaMemcpyAsync
void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, reinterpret_cast<cudaStream_t>(stream)));
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