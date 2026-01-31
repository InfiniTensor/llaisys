#pragma once
#include "llaisys/runtime.h"

#include "../utils.hpp"

namespace llaisys::device {
const LlaisysRuntimeAPI *getRuntimeAPI(llaisysDeviceType_t device_type);

const LlaisysRuntimeAPI *getUnsupportedRuntimeAPI();

namespace cpu {
const LlaisysRuntimeAPI *getRuntimeAPI();
}

#ifdef ENABLE_NVIDIA_API
namespace nvidia {
const LlaisysRuntimeAPI *getRuntimeAPI();
}
#endif
} // namespace llaisys::device
