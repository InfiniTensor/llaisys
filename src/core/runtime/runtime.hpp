#pragma once
#include "../core.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/allocator.hpp"
#include "../../device/device_resource.hpp"

namespace llaisys::core {
class Runtime {
private:
    llaisysDeviceType_t _device_type;
    int _device_id;
    const LlaisysRuntimeAPI *_api;
    MemoryAllocator *_allocator;
    llaisys::device::DeviceResource *_device_resource = nullptr;
    bool _is_active;
    void _activate();
    void _deactivate();
    llaisysStream_t _stream;
    Runtime(llaisysDeviceType_t device_type, int device_id);

public:
    friend class Context;

    ~Runtime();

    // Prevent copying
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    // Prevent moving
    Runtime(Runtime &&) = delete;
    Runtime &operator=(Runtime &&) = delete;

    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    bool isActive() const;

    const LlaisysRuntimeAPI *api() const;
    llaisys::device::DeviceResource *deviceResource() const { return _device_resource; }
    storage_t allocateDeviceStorage(size_t size);
    ;
    storage_t allocateHostStorage(size_t size);
    void freeStorage(Storage *storage);

    llaisysStream_t stream() const;
    void synchronize() const;
};
} // namespace llaisys::core
