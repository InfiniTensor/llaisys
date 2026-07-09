#include "llaisys/runtime.h"
#include "../core/context/context.hpp"
#include "../device/runtime_api.hpp"

// Llaisys API for setting context runtime.
__C void llaisysSetContextRuntime(llaisysDeviceType_t device_type, int device_id) {
    llaisys::core::context().setDevice(device_type, device_id);
}

// Llaisys API for getting the runtime APIs
__C const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t device_type) {
    return llaisys::device::getRuntimeAPI(device_type);
}

// Returns 1 if the library was compiled with support for the given device type, 0 otherwise.
__C int llaisysIsDeviceSupported(llaisysDeviceType_t device_type) {
    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return 1; // CPU is always supported
    case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_NVIDIA_API
        return 1;
#else
        return 0;
#endif
    default:
        return 0;
    }
}