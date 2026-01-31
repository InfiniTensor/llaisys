#include "runtime_api.hpp"

namespace llaisys::device {

int getDeviceCount() {
    return 0;
}

void setDevice(int) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void deviceSynchronize() {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

llaisysStream_t createStream() {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void destroyStream(llaisysStream_t stream) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}
void streamSynchronize(llaisysStream_t stream) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void *mallocDevice(size_t size) {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void freeDevice(void *ptr) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void *mallocHost(size_t size) {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void freeHost(void *ptr) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

static const LlaisysRuntimeAPI NOOP_RUNTIME_API = {
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

const LlaisysRuntimeAPI *getUnsupportedRuntimeAPI() {
    return &NOOP_RUNTIME_API;
}

const LlaisysRuntimeAPI *getRuntimeAPI(llaisysDeviceType_t device_type) {
    // Implement for all device types
    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return llaisys::device::cpu::getRuntimeAPI();
    case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_NVIDIA_API
        return llaisys::device::nvidia::getRuntimeAPI();
#else
        return getUnsupportedRuntimeAPI();
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
    }
}
} // namespace llaisys::device
