#include "context.hpp"
#include "../../tensor/tensor.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../../device/nvidia/nccl_context.hpp"
#endif
#include <thread>

namespace llaisys::core {

Context::Context() {
    // All device types, put CPU at the end
    std::vector<llaisysDeviceType_t> device_typs;
    for (int i = 1; i < LLAISYS_DEVICE_TYPE_COUNT; i++) {
        device_typs.push_back(static_cast<llaisysDeviceType_t>(i));
    }
    device_typs.push_back(LLAISYS_DEVICE_CPU);

    // Create runtimes for each device type.
    // Activate the first available device. If no other device is available, activate CPU runtime.
    for (auto device_type : device_typs) {
        const LlaisysRuntimeAPI *api_ = llaisysGetRuntimeAPI(device_type);
        int device_count = api_->get_device_count();
        std::vector<Runtime *> runtimes_(device_count);
        for (int device_id = 0; device_id < device_count; device_id++) {

            if (_current_runtime == nullptr) {
                auto runtime = new Runtime(device_type, device_id);
                runtime->_activate();
                runtimes_[device_id] = runtime;
                _current_runtime = runtime;
            }
        }
        _runtime_map[device_type] = runtimes_;
    }
}

Context::~Context() {
    finalizeDistributed();

    // Destroy current runtime first.
    delete _current_runtime;

    for (auto &runtime_entry : _runtime_map) {
        std::vector<Runtime *> runtimes = runtime_entry.second;
        for (auto runtime : runtimes) {
            if (runtime != nullptr && runtime != _current_runtime) {
                runtime->_activate();
                delete runtime;
            }
        }
        runtimes.clear();
    }
    _current_runtime = nullptr;
    _runtime_map.clear();
}

void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    // If doest not match the current runtime.
    if (_current_runtime == nullptr || _current_runtime->deviceType() != device_type || _current_runtime->deviceId() != device_id) {
        auto &runtimes = _runtime_map[device_type];
        CHECK_ARGUMENT((size_t)device_id < runtimes.size() && device_id >= 0, "invalid device id");
        if (_current_runtime != nullptr) {
            _current_runtime->_deactivate();
        }
        if (runtimes[device_id] == nullptr) {
            runtimes[device_id] = new Runtime(device_type, device_id);
        }
        runtimes[device_id]->_activate();
        _current_runtime = runtimes[device_id];
    }
}

Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is activated, please call setDevice() first.");
    return *_current_runtime;
}

void Context::initDistributed(int rank, int world_size) {
#ifdef ENABLE_NVIDIA_API
    setDevice(LLAISYS_DEVICE_NVIDIA, rank);
    _dist_context = std::make_unique<llaisys::device::nvidia::NcclContext>(rank, world_size, runtime().deviceId());
    return;
#else
    (void)rank;
    (void)world_size;
    EXCEPTION_UNSUPPORTED_DEVICE;
#endif
}

void Context::finalizeDistributed() {
#ifdef ENABLE_NVIDIA_API
    _dist_context.reset();
#endif
}

bool Context::distributedInitialized() const {
#ifdef ENABLE_NVIDIA_API
    return _dist_context != nullptr;
#else
    return false;
#endif
}

int Context::distributedRank() const {
#ifdef ENABLE_NVIDIA_API
    ASSERT(_dist_context != nullptr, "Distributed context is not initialized.");
    return _dist_context->rank();
#else
    EXCEPTION_UNSUPPORTED_DEVICE;
#endif
}

int Context::distributedWorldSize() const {
#ifdef ENABLE_NVIDIA_API
    ASSERT(_dist_context != nullptr, "Distributed context is not initialized.");
    return _dist_context->worldSize();
#else
    EXCEPTION_UNSUPPORTED_DEVICE;
#endif
}

void Context::allReduce(const tensor_t &tensor) {
    ASSERT(tensor != nullptr, "allReduce tensor must not be null.");
    ASSERT(tensor->isContiguous(), "allReduce requires a contiguous tensor.");
    ASSERT(tensor->deviceType() == LLAISYS_DEVICE_NVIDIA, "Distributed collectives currently support NVIDIA tensors only.");
#ifdef ENABLE_NVIDIA_API
    ASSERT(_dist_context != nullptr, "Distributed context is not initialized.");
    setDevice(tensor->deviceType(), tensor->deviceId());
    ASSERT(runtime().deviceId() == _dist_context->deviceId(), "Tensor device does not match distributed communicator device.");
    _dist_context->allReduce(tensor->data(), tensor->numel(), tensor->dtype(), runtime().stream());
    runtime().synchronize();
#else
    EXCEPTION_UNSUPPORTED_DEVICE;
#endif
}

tensor_t Context::allGather(const tensor_t &tensor) {
    ASSERT(tensor != nullptr, "allGather tensor must not be null.");
    ASSERT(tensor->isContiguous(), "allGather requires a contiguous tensor.");
    ASSERT(tensor->deviceType() == LLAISYS_DEVICE_NVIDIA, "Distributed collectives currently support NVIDIA tensors only.");
#ifdef ENABLE_NVIDIA_API
    ASSERT(_dist_context != nullptr, "Distributed context is not initialized.");
    setDevice(tensor->deviceType(), tensor->deviceId());
    ASSERT(runtime().deviceId() == _dist_context->deviceId(), "Tensor device does not match distributed communicator device.");

    std::vector<size_t> gathered_shape;
    gathered_shape.reserve(tensor->ndim() + 1);
    gathered_shape.push_back(static_cast<size_t>(_dist_context->worldSize()));
    gathered_shape.insert(gathered_shape.end(), tensor->shape().begin(), tensor->shape().end());

    auto gathered = llaisys::Tensor::create(gathered_shape, tensor->dtype(), tensor->deviceType(), tensor->deviceId());
    _dist_context->allGather(tensor->data(), gathered->data(), tensor->numel(), tensor->dtype(), runtime().stream());
    runtime().synchronize();
    return gathered;
#else
    EXCEPTION_UNSUPPORTED_DEVICE;
#endif
}

void Context::broadcast(const tensor_t &tensor, int root) {
    ASSERT(tensor != nullptr, "broadcast tensor must not be null.");
    ASSERT(tensor->isContiguous(), "broadcast requires a contiguous tensor.");
    ASSERT(tensor->deviceType() == LLAISYS_DEVICE_NVIDIA, "Distributed collectives currently support NVIDIA tensors only.");
#ifdef ENABLE_NVIDIA_API
    ASSERT(_dist_context != nullptr, "Distributed context is not initialized.");
    setDevice(tensor->deviceType(), tensor->deviceId());
    ASSERT(runtime().deviceId() == _dist_context->deviceId(), "Tensor device does not match distributed communicator device.");
    _dist_context->broadcast(tensor->data(), tensor->numel(), tensor->dtype(), root, runtime().stream());
    runtime().synchronize();
#else
    (void)root;
    EXCEPTION_UNSUPPORTED_DEVICE;
#endif
}

void Context::barrier() {
#ifdef ENABLE_NVIDIA_API
    ASSERT(_dist_context != nullptr, "Distributed context is not initialized.");
    setDevice(LLAISYS_DEVICE_NVIDIA, _dist_context->deviceId());
    _dist_context->barrier(runtime().stream());
    runtime().synchronize();
#else
    EXCEPTION_UNSUPPORTED_DEVICE;
#endif
}

// Global API to get thread-local context.
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}

} // namespace llaisys::core
