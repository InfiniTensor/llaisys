#pragma once

#include "llaisys.h"

#include "../core.hpp"

#include "../runtime/runtime.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace llaisys {
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;
}

#ifdef ENABLE_NVIDIA_API
namespace llaisys::device::nvidia {
class NcclContext;
}
#endif

namespace llaisys::core {
class Context {
private:
    std::unordered_map<llaisysDeviceType_t, std::vector<Runtime *>> _runtime_map;
    Runtime *_current_runtime = nullptr;
#ifdef ENABLE_NVIDIA_API
    std::unique_ptr<llaisys::device::nvidia::NcclContext> _dist_context;
#endif
    Context();

public:
    ~Context();

    // Prevent copy
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    // Prevent move
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    void setDevice(llaisysDeviceType_t device_type, int device_id);
    Runtime &runtime();

    void initDistributed(int rank, int world_size);
    void finalizeDistributed();
    bool distributedInitialized() const;
    int distributedRank() const;
    int distributedWorldSize() const;

    void allReduce(const tensor_t &tensor);
    tensor_t allGather(const tensor_t &tensor);
    void broadcast(const tensor_t &tensor, int root);
    void barrier();

    friend Context &context();
};
} // namespace llaisys::core
