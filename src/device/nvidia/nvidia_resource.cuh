#pragma once

#include "../device_resource.hpp"
#include <cublas_v2.h>

#include <memory>
#include <mutex>
#include <unordered_map>

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
private:
    cublasHandle_t _cublas_handle;

public:
    Resource(int device_id);
    ~Resource();

    cublasHandle_t cublasHandle() const;
};

Resource &resource(int device_id);
} // namespace llaisys::device::nvidia
