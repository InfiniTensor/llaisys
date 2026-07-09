#include "../runtime_api.hpp"

#include "metax_utils.cuh"

#include <cstring>

namespace llaisys::device::metax {

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    METAX_CHECK(mcGetDeviceCount(&count));
    return count;
}

void setDevice(int device_id) {
    METAX_CHECK(mcSetDevice(device_id));
}

void deviceSynchronize() {
    METAX_CHECK(mcDeviceSynchronize());
}

llaisysStream_t createStream() {
    mcStream_t stream = nullptr;
    METAX_CHECK(mcStreamCreateWithFlags(&stream, mcStreamNonBlocking));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    METAX_CHECK(mcStreamDestroy(reinterpret_cast<mcStream_t>(stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    METAX_CHECK(mcStreamSynchronize(reinterpret_cast<mcStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    METAX_CHECK(mcMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        METAX_CHECK(mcFree(ptr));
    }
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    METAX_CHECK(mcMallocHost(&ptr, size, mcMallocHostDefault));
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        METAX_CHECK(mcFreeHost(ptr));
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    if (kind == LLAISYS_MEMCPY_H2H) {
        std::memcpy(dst, src, size);
        return;
    }
    // 公开 Runtime API 没有显式 stream 参数。这里先同步设备，
    // 让 Python/PyTorch 侧在同步拷贝后立刻看到一致数据。
    METAX_CHECK(mcDeviceSynchronize());
    METAX_CHECK(mcMemcpy(dst, src, size, to_mc_memcpy_kind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    if (kind == LLAISYS_MEMCPY_H2H) {
        std::memcpy(dst, src, size);
        return;
    }
    METAX_CHECK(mcMemcpyAsync(dst, src, size, to_mc_memcpy_kind(kind), reinterpret_cast<mcStream_t>(stream)));
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
