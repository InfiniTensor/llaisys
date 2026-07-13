#include "../runtime_api.hpp"
#include "../../utils/check.hpp"

#include <cuda_runtime.h>
#include <iostream>

#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (0)

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
   int count=0;
   if(cudaGetDeviceCount(&count)!=cudaSuccess)
   {
     return 0;
   }
   return count;
}

void setDevice(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return static_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
}
void streamSynchronize(llaisysStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr=nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if(ptr!=nullptr)
    {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void *mallocHost(size_t size) {
   void *ptr=nullptr;
   CUDA_CHECK(cudaMallocHost(&ptr, size));
   return ptr;
}

void freeHost(void *ptr) {
    if(ptr!=nullptr)
    {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

static cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
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
            return cudaMemcpyDefault;
    }
}
void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind,llaisysStream_t stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), cuda_stream));
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
