#pragma once

#include "../device_resource.hpp"

#include <mcblas/mcblas.h>

namespace llaisys::device::metax {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
};

mcblasHandle_t get_mcblas_handle(int device_id, void *stream);
} // namespace llaisys::device::metax
