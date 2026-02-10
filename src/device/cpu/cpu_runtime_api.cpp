#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

namespace llaisys::device::cpu {

namespace runtime_api {
int getDeviceCount() {
    return 1;
}

void setDevice(int) {
    // do nothing
}

void deviceSynchronize() {
    // do nothing
}

llaisysStream_t createStream() {
    return (llaisysStream_t)0; // null stream
}

void destroyStream(llaisysStream_t stream) {
    // do nothing
}
void streamSynchronize(llaisysStream_t stream) {
    // do nothing
}

void *mallocDevice(size_t size) {
    return std::malloc(size);
}

void freeDevice(void *ptr) {
    std::free(ptr);
}

void *mallocHost(size_t size) {
    return mallocDevice(size);
}

void freeHost(void *ptr) {
    freeDevice(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    std::memcpy(dst, src, size);
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    memcpySync(dst, src, size, kind);
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
} // namespace llaisys::device::cpu
