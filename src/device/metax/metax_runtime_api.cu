#include "../runtime_api.hpp"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

namespace llaisys::device::metax {

namespace runtime_api {

constexpr size_t ALIGNMENT = 64;

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
    size_t aligned_size = ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    cudaMalloc(&ptr, aligned_size > 0 ? aligned_size : ALIGNMENT);
    return ptr;
}

void freeDevice(void *ptr) {
    cudaFree(ptr);
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    size_t aligned_size = ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    posix_memalign(&ptr, ALIGNMENT, aligned_size > 0 ? aligned_size : ALIGNMENT);
    return ptr;
}

void freeHost(void *ptr) {
    free(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    bool dst_aligned = (reinterpret_cast<uintptr_t>(dst) % ALIGNMENT) == 0;
    bool src_aligned = (reinterpret_cast<uintptr_t>(src) % ALIGNMENT) == 0;
    bool size_aligned = (size % ALIGNMENT) == 0;
    
    // For MetaX, we need to handle misaligned D2D copies specially
    // because mcMemcpy requires both source and destination to be aligned
    if (kind == LLAISYS_MEMCPY_D2D && (!dst_aligned || !src_aligned || !size_aligned)) {
        // Use a kernel to copy data element by element to avoid alignment issues
        // This is slower but works for misaligned addresses
        void *temp_buf = nullptr;
        size_t aligned_size = ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
        cudaMalloc(&temp_buf, aligned_size);
        
        if (temp_buf) {
            // D2H to temp buffer (temp_buf is aligned from cudaMalloc)
            cudaError_t err = cudaMemcpy(temp_buf, src, size, cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                // H2D from temp buffer to destination
                err = cudaMemcpy(dst, temp_buf, size, cudaMemcpyHostToDevice);
            }
            if (err != cudaSuccess) {
                fprintf(stderr, "[MetaX] D2D copy via temp buffer failed: %s\n", cudaGetErrorString(err));
            }
            cudaFree(temp_buf);
        }
        return;
    }
    
    if (dst_aligned && src_aligned && size_aligned) {
        cudaError_t err = cudaMemcpy(dst, src, size, convertMemcpyKind(kind));
        if (err != cudaSuccess) {
            fprintf(stderr, "[MetaX] cudaMemcpy failed: %s (dst=%p, src=%p, size=%zu)\n", 
                    cudaGetErrorString(err), dst, src, size);
        }
    } else {
        void *aligned_buf = nullptr;
        size_t aligned_size = ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
        posix_memalign(&aligned_buf, ALIGNMENT, aligned_size);
        
        cudaError_t err = cudaSuccess;
        switch (kind) {
            case LLAISYS_MEMCPY_H2D:
                std::memcpy(aligned_buf, src, size);
                err = cudaMemcpy(dst, aligned_buf, size, cudaMemcpyHostToDevice);
                break;
            case LLAISYS_MEMCPY_D2H:
                err = cudaMemcpy(aligned_buf, src, size, cudaMemcpyDeviceToHost);
                std::memcpy(dst, aligned_buf, size);
                break;
            case LLAISYS_MEMCPY_D2D:
                // This should not happen due to the check above, but handle it anyway
                err = cudaMemcpy(aligned_buf, src, size, cudaMemcpyDeviceToHost);
                if (err == cudaSuccess) {
                    err = cudaMemcpy(dst, aligned_buf, size, cudaMemcpyHostToDevice);
                }
                break;
            default:
                std::memcpy(dst, src, size);
                break;
        }
        
        if (err != cudaSuccess) {
            fprintf(stderr, "[MetaX] cudaMemcpy (fallback) failed: %s (dst=%p, src=%p, size=%zu)\n", 
                    cudaGetErrorString(err), dst, src, size);
        }
        
        free(aligned_buf);
    }
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
} // namespace llaisys::device::metax
