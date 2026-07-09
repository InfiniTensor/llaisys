#pragma once

#include "../device_resource.hpp"

#include <cublas_v2.h>

namespace llaisys::device::nvidia {

// 获取当前设备的 cuBLAS handle（懒初始化，按设备 ID 缓存）
cublasHandle_t getCublasHandle();

class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
};

} // namespace llaisys::device::nvidia
