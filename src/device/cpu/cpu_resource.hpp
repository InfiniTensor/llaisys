#pragma once

#include "../device_resource.hpp"

namespace llaisys::device::cpu {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource();
    ~Resource() = default;
};
} // namespace llaisys::device::cpu