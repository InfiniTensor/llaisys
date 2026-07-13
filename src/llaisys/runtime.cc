#include "llaisys/runtime.h"
#include "llaisys_tensor.hpp"
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

__C void llaisysInitDistributed(int rank, int world_size) {
    llaisys::core::context().initDistributed(rank, world_size);
}

__C void llaisysFinalizeDistributed() {
    llaisys::core::context().finalizeDistributed();
}

__C uint8_t llaisysDistributedIsInitialized() {
    return static_cast<uint8_t>(llaisys::core::context().distributedInitialized());
}

__C int llaisysDistributedRank() {
    return llaisys::core::context().distributedRank();
}

__C int llaisysDistributedWorldSize() {
    return llaisys::core::context().distributedWorldSize();
}

__C void llaisysDistAllReduce(llaisysTensor_t tensor) {
    llaisys::core::context().allReduce(tensor->tensor);
}

__C llaisysTensor_t llaisysDistAllGather(llaisysTensor_t tensor) {
    return new LlaisysTensor{llaisys::core::context().allGather(tensor->tensor)};
}

__C void llaisysDistBroadcast(llaisysTensor_t tensor, int root) {
    llaisys::core::context().broadcast(tensor->tensor, root);
}

__C void llaisysDistBarrier() {
    llaisys::core::context().barrier();
}