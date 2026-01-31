#pragma once

#include "../device_resource.hpp"

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
};
} // namespace llaisys::device::nvidia
