#pragma once

#include "../device_resource.hpp"

#include <cublas_v2.h>

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
};

cublasHandle_t get_cublas_handle(int device_id, void *stream);
} // namespace llaisys::device::nvidia
