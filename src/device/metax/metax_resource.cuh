#pragma once

#include "../device_resource.hpp"

// Forward declaration for mcblas types
typedef struct mcblasContext* mcblasHandle_t;

namespace llaisys::device::metax {

// Singleton mcblas handle for lazy initialization
class McblasHandle {
public:
    static mcblasHandle_t& get();
    
private:
    McblasHandle();
    ~McblasHandle();
    mcblasHandle_t handle_;
    
    McblasHandle(const McblasHandle&) = delete;
    McblasHandle& operator=(const McblasHandle&) = delete;
};

class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
};

} // namespace llaisys::device::metax
