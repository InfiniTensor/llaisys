#include "context.hpp"
#include "../../utils.hpp"
#include <mutex>
#include <thread>

namespace llaisys::core {

namespace {
// 进程内共享的 Runtime 池，保证多线程（如 worker）与主线程使用同一 CUDA 上下文，避免模型在主线程分配、在 worker 访问时 segfault。
struct GlobalRuntimePool {
    std::mutex mutex;
    std::unordered_map<llaisysDeviceType_t, std::vector<Runtime *>> pool;
    bool initialized = false;

    ~GlobalRuntimePool() {
        for (auto &entry : pool) {
            for (Runtime *r : entry.second) {
                if (r != nullptr) {
                    r->deactivateForShutdown();
                    delete r;
                }
            }
        }
    }
} g_runtime_pool;
} // namespace

Context::Context() {
    std::lock_guard<std::mutex> lock(g_runtime_pool.mutex);
    if (!g_runtime_pool.initialized) {
        std::vector<llaisysDeviceType_t> device_typs;
        for (int i = 1; i < LLAISYS_DEVICE_TYPE_COUNT; i++)
            device_typs.push_back(static_cast<llaisysDeviceType_t>(i));
        device_typs.push_back(LLAISYS_DEVICE_CPU);

        Runtime *first = nullptr;
        for (auto device_type : device_typs) {
            const LlaisysRuntimeAPI *api_ = llaisysGetRuntimeAPI(device_type);
            int device_count = api_->get_device_count();
            std::vector<Runtime *> runtimes_(device_count, nullptr);
            for (int device_id = 0; device_id < device_count; device_id++) {
                auto *r = new Runtime(device_type, device_id);
                runtimes_[device_id] = r;
                if (first == nullptr) {
                    r->_activate();
                    first = r;
                }
            }
            g_runtime_pool.pool[device_type] = runtimes_;
        }
        g_runtime_pool.initialized = true;
    }
    _runtime_map = g_runtime_pool.pool;
    _current_runtime = nullptr;
    // 默认激活 CPU runtime，避免未调用 setDevice 的代码路径（如首次 Tensor::create）触发 runtime() 断言
    auto it = _runtime_map.find(LLAISYS_DEVICE_CPU);
    if (it != _runtime_map.end() && !it->second.empty() && it->second[0] != nullptr) {
        it->second[0]->_activate();
        _current_runtime = it->second[0];
    }
}

Context::~Context() {
    if (_current_runtime != nullptr)
        _current_runtime->_deactivate();
    _current_runtime = nullptr;
    _runtime_map.clear();
}

void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    // If doest not match the current runtime.
    if (_current_runtime == nullptr || _current_runtime->deviceType() != device_type || _current_runtime->deviceId() != device_id) {
        auto runtimes = _runtime_map[device_type];
        if (runtimes.empty()) {
            if (device_type == LLAISYS_DEVICE_NVIDIA)
                CHECK_ARGUMENT(false, "no NVIDIA GPUs available (get_device_count() returned 0). Use --device cpu or check CUDA/driver.");
            else
                CHECK_ARGUMENT(false, "no devices available for this device type (get_device_count() returned 0).");
        }
        CHECK_ARGUMENT((size_t)device_id < runtimes.size() && device_id >= 0, "invalid device id");
        if (_current_runtime != nullptr) {
            _current_runtime->_deactivate();
        }
        // Runtime 必须来自全局池，保证多线程与主线程使用同一 CUDA 上下文；不得在此处 new 仅写入本线程 _runtime_map 副本，否则会泄漏且破坏设备一致性。
        CHECK_ARGUMENT(runtimes[device_id] != nullptr, "runtime for device type/id not found in pool; ensure Context is used after global pool init.");
        runtimes[device_id]->_activate();
        _current_runtime = runtimes[device_id];
    }
}

Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is activated, please call setDevice() first.");
    return *_current_runtime;
}

// Global API to get thread-local context.
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}

} // namespace llaisys::core
