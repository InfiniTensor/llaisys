#include "op.hpp"
#include "../../utils.hpp"
#include "./cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 1. 基础校验
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    // 2. 形状校验
    CHECK_ARGUMENT(in->ndim() == 2, "RMSNorm: input must be 2D");
    CHECK_ARGUMENT(weight->ndim() == 1, "RMSNorm: weight must be 1D");
    
    size_t d = in->shape().back(); // 获取输入张量 in 最后一个维度的大小。
    size_t M = in->numel() / d; //  计算输入张量中总行数（或者说需要独立进行归一化操作的向量个数）。

    CHECK_ARGUMENT(weight->shape()[0] == d, "RMSNorm: weight size mismatch");
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    // 3. 内存连续性校验
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "RMSNorm: tensors must be contiguous");

    // 4. 设备分发
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), M, d, eps);
        return;
    }

    // GPU 分支待实现
    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops
