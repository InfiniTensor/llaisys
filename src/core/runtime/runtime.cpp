#include "runtime.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/naive_allocator.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../../device/nvidia/nvidia_resource.cuh"
#endif

namespace llaisys::core {
Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type), _device_id(device_id), _is_active(false) {
    _api = llaisys::device::getRuntimeAPI(_device_type);
    _stream = _api->create_stream();
    _allocator = new allocators::NaiveAllocator(_api);
    if (device_type == LLAISYS_DEVICE_NVIDIA) {
#ifdef ENABLE_NVIDIA_API
    // 只有在 CUDA 工具链存在的环境下，才会被编译进二进制文件
    _device_resource = new llaisys::device::nvidia::Resource(device_id);
#else
    // 在无 CUDA 环境下抛出绝对明确的运行时异常
    throw std::runtime_error("Llaisys Runtime Error: The framework was compiled without NVIDIA backend support. No CUDA toolkit detected during build.");
#endif
}
}

Runtime::~Runtime() {
    if (!_is_active) {
        std::cerr << "Mallicious destruction of inactive runtime." << std::endl;
    }
    delete _allocator;
    _allocator = nullptr;
    _api->destroy_stream(_stream);
    _api = nullptr;
    if (_device_resource) {
        delete _device_resource;
        _device_resource = nullptr;
    }
}

void Runtime::_activate() {
    _api->set_device(_device_id);
    _is_active = true;
}

void Runtime::_deactivate() {
    _is_active = false;
}

bool Runtime::isActive() const {
    return _is_active;
}

llaisysDeviceType_t Runtime::deviceType() const {
    return _device_type;
}

int Runtime::deviceId() const {
    return _device_id;
}

const LlaisysRuntimeAPI *Runtime::api() const {
    return _api;
}

storage_t Runtime::allocateDeviceStorage(size_t size) {
    return std::shared_ptr<Storage>(new Storage(_allocator->allocate(size), size, *this, false));
}

storage_t Runtime::allocateHostStorage(size_t size) {
    return std::shared_ptr<Storage>(new Storage((std::byte *)_api->malloc_host(size), size, *this, true));
}

void Runtime::freeStorage(Storage *storage) {
    if (storage->isHost()) {
        _api->free_host(storage->memory());
    } else {
        _allocator->release(storage->memory());
    }
}

llaisysStream_t Runtime::stream() const {
    return _stream;
}

void Runtime::synchronize() const {
    _api->stream_synchronize(_stream);
}

} // namespace llaisys::core
