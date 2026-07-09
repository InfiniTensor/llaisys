#include "../runtime_api.hpp"

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

// CUDA error check helper
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t _err = (call);                                  \
        if (_err != cudaSuccess) {                                  \
            std::ostringstream _oss;                                \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                 << " : " << cudaGetErrorString(_err);              \
            throw std::runtime_error(_oss.str());                   \
        }                                                           \
    } while (0)

namespace llaisys::device::nvidia {

namespace runtime_api {

int getDeviceCount() {
    int count = 0;
    // Force CUDA runtime initialization before querying device count.
    // In some container environments (e.g. DSW, Docker without proper NVIDIA hooks)
    // cudaGetDeviceCount() returns 0 until the runtime is explicitly initialized.
    // cudaFree(nullptr) is a guaranteed no-op that triggers that initialization;
    // its return value may be non-success if no device is present, so we ignore it.
    (void)cudaFree(nullptr);
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
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    // cudaMallocHost provides page-locked memory for faster H2D/D2H transfers
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

static cudaMemcpyKind tocudaMemcpyKind(llaisysMemcpyKind_t kind) {
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
        throw std::invalid_argument("memcpy: unknown llaisysMemcpyKind_t");
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, tocudaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind,
                 llaisysStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, tocudaMemcpyKind(kind),
                               static_cast<cudaStream_t>(stream)));
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
