#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <vector>

namespace llaisys::device::nvidia {

namespace runtime_api {
namespace {
void check_cuda(cudaError_t err, const char *msg);

std::vector<int> &available_devices() {
    static std::vector<int> devices;
    static bool initialized = false;
    if (initialized) {
        return devices;
    }
    initialized = true;

    int ndev = 0;
    cudaError_t err = cudaGetDeviceCount(&ndev);
    if (err == cudaErrorNoDevice) {
        return devices;
    }
    check_cuda(err, "cudaGetDeviceCount");

    for (int dev = 0; dev < ndev; ++dev) {
        if (cudaSetDevice(dev) != cudaSuccess) {
            (void)cudaGetLastError();
            continue;
        }
        // Warm up context creation to filter out temporarily unavailable devices.
        if (cudaFree(nullptr) != cudaSuccess) {
            (void)cudaGetLastError();
            continue;
        }
        devices.push_back(dev);
    }
    return devices;
}

void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "[CUDA] " << msg << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

cudaMemcpyKind to_cuda_memcpy_kind(llaisysMemcpyKind_t kind) {
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
} // namespace

int getDeviceCount() {
    return static_cast<int>(available_devices().size());
}

void setDevice(int device_id) {
    auto &devices = available_devices();
    if (device_id < 0 || static_cast<size_t>(device_id) >= devices.size()) {
        throw std::invalid_argument("invalid nvidia device id");
    }
    check_cuda(cudaSetDevice(devices[static_cast<size_t>(device_id)]), "cudaSetDevice");
    check_cuda(cudaFree(nullptr), "cudaFree(warmup)");
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
    if (!stream) {
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
    if (!ptr) {
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
    if (!ptr) {
        return;
    }
    check_cuda(cudaFreeHost(ptr), "cudaFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    check_cuda(cudaMemcpy(dst, src, size, to_cuda_memcpy_kind(kind)), "cudaMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    check_cuda(cudaMemcpyAsync(dst,
                               src,
                               size,
                               to_cuda_memcpy_kind(kind),
                               reinterpret_cast<cudaStream_t>(stream)),
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
