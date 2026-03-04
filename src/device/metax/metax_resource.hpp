#pragma once

#include "../device_resource.hpp"

namespace llaisys::device::metax {
class Resource : public llaisys::device::DeviceResource {
public:
    explicit Resource(int device_id);
    ~Resource() = default;
};
} // namespace llaisys::device::metax
