#include "metax_resource.cuh"
#include "metax_utils.cuh"

#include <unordered_map>

namespace llaisys::device::metax {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_METAX, device_id) {}
Resource::~Resource() = default;

mcblasHandle_t get_mcblas_handle(int device_id, void *stream) {
    thread_local std::unordered_map<int, mcblasHandle_t> handles;
    auto iter = handles.find(device_id);
    if (iter == handles.end()) {
        // mcBLAS 句柄要求当前线程已经切到目标设备，否则 create 会直接失败。
        METAX_CHECK(mcSetDevice(device_id));
        mcblasHandle_t handle = nullptr;
        MCBLAS_CHECK(mcblasCreate(&handle));
        iter = handles.emplace(device_id, handle).first;
    }
    MCBLAS_CHECK(mcblasSetStream(iter->second, reinterpret_cast<mcStream_t>(stream)));
    return iter->second;
}

} // namespace llaisys::device::metax
