#include "cpu_resource.hpp"

namespace llaisys::device::cpu {
Resource::Resource() : llaisys::device::DeviceResource(LLAISYS_DEVICE_CPU, 0) {}
} // namespace llaisys::device::cpu
