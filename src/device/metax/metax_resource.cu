#include "metax_resource.cuh"

#include <cublas_v2.h>

namespace llaisys::device::metax {

// McblasHandle implementation
mcblasHandle_t& McblasHandle::get() {
    static McblasHandle instance;
    return instance.handle_;
}

McblasHandle::McblasHandle() {
    mcblasCreate(&handle_);
}

McblasHandle::~McblasHandle() {
    if (handle_) {
        mcblasDestroy(handle_);
    }
}

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_METAX, device_id) {}

Resource::~Resource() = default;

} // namespace llaisys::device::metax
