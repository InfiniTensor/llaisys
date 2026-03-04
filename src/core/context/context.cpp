#include "context.hpp"
#include "../../utils.hpp"
#include <thread>

namespace llaisys::core {
    
// 构造函数：初始化运行时环境
Context::Context() {
    // 所有设备类型，将 CPU 放在最后尝试
    std::vector<llaisysDeviceType_t> device_typs;
    for (int i = 1; i < LLAISYS_DEVICE_TYPE_COUNT; i++) {
        device_typs.push_back(static_cast<llaisysDeviceType_t>(i));
    }
    device_typs.push_back(LLAISYS_DEVICE_CPU);

    // 为每种设备类型创建运行时实例。
    // 激活首个可用的设备；若无其他设备可用，则激活 CPU 运行时。
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

// 析构函数：释放上下文及其管理的所有运行时资源
Context::~Context() {
    // 先销毁当前激活的运行时。
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

// 切换当前使用的设备
void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    // 如果与当前运行时不匹配，则切换
    if (_current_runtime == nullptr || _current_runtime->deviceType() != device_type || _current_runtime->deviceId() != device_id) {
        auto runtimes = _runtime_map[device_type];
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

// 获取当前激活的运行时引用
Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is activated, please call setDevice() first.");
    return *_current_runtime;
}

// 全局接口：获取线程本地的上下文实例
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}

} // namespace llaisys::core