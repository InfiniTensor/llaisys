#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif

namespace llaisys::ops {
/**
 * RMS Norm (Root Mean Square Layer Normalization)
 *
 * 公式: out[i][j] = (in[i][j] / RMS(in[i])) * weight[j]
 *       其中 RMS(x) = sqrt(mean(x^2) + eps)
 *
 * 参数说明:
 *   - batch: 需要独立归一化的向量数量，等于 in.numel() / hidden_size
 *            例如 in 形状为 [2, 3, 768]，则 batch = 2*3 = 6
 *   - hidden_size: 每个向量的特征维度，即 in 的最后一个维度
 *                  RMS 在此维度上计算，weight 的长度也等于 hidden_size
 *
 * 张量形状:
 *   - in/out: [..., hidden_size]  (任意前导维度，最后一维是 hidden_size)
 *   - weight: [hidden_size]       (所有向量共享同一组 weight)
 */
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RmsNorm: all tensors must be contiguous.");

    ASSERT(in->ndim() >= 1, "RmsNorm: in must have at least 1 dimension.");
    ASSERT(weight->ndim() == 1, "RmsNorm: weight must be 1D.");

    // hidden_size: in 的最后一个维度，RMS 在此维度上计算
    size_t hidden_size = in->shape().back();
    // batch: 独立归一化的向量数量 (前导维度的乘积)
    size_t batch = in->numel() / hidden_size;
    
    ASSERT(weight->shape()[0] == hidden_size, "RmsNorm: weight.shape[0] must be equal to in's last dimension.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), batch, hidden_size, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), batch, hidden_size, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), batch, hidden_size, eps,
                                llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
