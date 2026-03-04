#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#include "nvidia/argmax_nvidia.cuh"
#ifdef ENABLE_METAX_API
#include "metax/argmax_metax.hpp"
#endif
#include "llaisys.h"

// 参数检验+设备分发
namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 1. 检测张量所在设备
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // 2. 检测张量形状，目前仅支持一维张量
    CHECK_ARGUMENT(vals->ndim() == 1, "vals only support 1D tensor for now");
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_idx->numel() == 1, "max_idx should be a single element");
    CHECK_ARGUMENT(max_val->ndim() == 1 && max_val->numel() == 1, "max_val should be a single element");
    
    // 3. 检测张量数据类型，目前仅支持Int64类型，max_index与pytorch对齐，使用64位
    CHECK_SAME_DTYPE(max_idx->dtype(), LLAISYS_DTYPE_I64);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    // 4. 检测张量是否连续
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "max_idx, max_val and vals must be contiguous");

    // 5. 设置上下文，切换当前计算上下文到张量所在设备
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(reinterpret_cast<int64_t*>(max_idx->data()), max_val->data(), vals->data(), 
                     vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(reinterpret_cast<int64_t*>(max_idx->data()), reinterpret_cast<std::byte*>(max_val->data()), reinterpret_cast<const std::byte*>(vals->data()), 
                     vals->dtype(), vals->numel());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::argmax(reinterpret_cast<int64_t*>(max_idx->data()),
                             reinterpret_cast<std::byte*>(max_val->data()),
                             reinterpret_cast<const std::byte*>(vals->data()),
                             vals->dtype(),
                             vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}
} // namespace llaisys::ops
