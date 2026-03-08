#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cmath>

#include <vector>

namespace {
template <typename T>
void swiglu_impl(T *out_ptr, const T *gate_ptr, const T *up_ptr, size_t total_size) {
    // out[i] = up[i] * gate[i] / (1 + exp(-gate[i]))
    // 这是门控线性单元的一个变体
    for (size_t i = 0; i < total_size; ++i) {
        // 转换为float计算，保证数值精度
        float gate_val = llaisys::utils::cast<float>(gate_ptr[i]);
        float up_val = llaisys::utils::cast<float>(up_ptr[i]);

        // 计算 gate / (1 + exp(-gate))
        // 处理数值稳定性：避免exp(-gate_val)爆炸
        float glu_val;
        if (gate_val >= 50.0f) {
            // 当gate_val >= 50时，exp(-gate_val) ≈ 0，所以 gate / (1 + 0) ≈ gate
            glu_val = gate_val;
        } else if (gate_val <= -50.0f) {
            // 当gate_val <= -50时，exp(-gate_val)很大，gate / (1 + exp(-gate)) ≈ 0
            glu_val = 0.0f;
        } else {
            // 一般情况：gate / (1 + exp(-gate))
            glu_val = gate_val / (1.0f + std::exp(-gate_val));
        }

        // 计算输出：up * glu
        out_ptr[i] = llaisys::utils::cast<T>(up_val * glu_val);
    }
}
} // namespace

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 基本验证：设备类型、数据类型、连续性
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: all tensors must be contiguous.");

    // 维度和形状检查
    CHECK_ARGUMENT(out->ndim() == gate->ndim() && gate->ndim() == up->ndim(),
                   "SwiGLU: out, gate, and up must have the same number of dimensions.");
    CHECK_ARGUMENT(out->shape() == gate->shape() && gate->shape() == up->shape(),
                   "SwiGLU: out, gate, and up must have the same shape.");

    size_t total_size = out->numel();
    auto type = out->dtype();

    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        auto &ctx = llaisys::core::context();
        ctx.setDevice(out->deviceType(), out->deviceId());
        const auto *api = ctx.runtime().api();

        const size_t out_bytes = out->numel() * out->elementSize();
        const size_t gate_bytes = gate->numel() * gate->elementSize();
        const size_t up_bytes = up->numel() * up->elementSize();
        std::vector<std::byte> h_out(out_bytes), h_gate(gate_bytes), h_up(up_bytes);
        api->memcpy_sync(h_gate.data(), gate->data(), gate_bytes, LLAISYS_MEMCPY_D2H);
        api->memcpy_sync(h_up.data(), up->data(), up_bytes, LLAISYS_MEMCPY_D2H);

        switch (type) {
        case LLAISYS_DTYPE_F32:
            swiglu_impl(reinterpret_cast<float *>(h_out.data()),
                        reinterpret_cast<const float *>(h_gate.data()),
                        reinterpret_cast<const float *>(h_up.data()),
                        total_size);
            break;
        case LLAISYS_DTYPE_F16:
            swiglu_impl(reinterpret_cast<llaisys::fp16_t *>(h_out.data()),
                        reinterpret_cast<const llaisys::fp16_t *>(h_gate.data()),
                        reinterpret_cast<const llaisys::fp16_t *>(h_up.data()),
                        total_size);
            break;
        case LLAISYS_DTYPE_BF16:
            swiglu_impl(reinterpret_cast<llaisys::bf16_t *>(h_out.data()),
                        reinterpret_cast<const llaisys::bf16_t *>(h_gate.data()),
                        reinterpret_cast<const llaisys::bf16_t *>(h_up.data()),
                        total_size);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }

        api->memcpy_sync(out->data(), h_out.data(), out_bytes, LLAISYS_MEMCPY_H2D);
        return;
    }

    // 当前仅支持CPU设备
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    // dtype分发：支持F32、F16、BF16
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl(reinterpret_cast<float *>(out->data()),
                           reinterpret_cast<const float *>(gate->data()),
                           reinterpret_cast<const float *>(up->data()),
                           total_size);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                           reinterpret_cast<const llaisys::fp16_t *>(gate->data()),
                           reinterpret_cast<const llaisys::fp16_t *>(up->data()),
                           total_size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                           reinterpret_cast<const llaisys::bf16_t *>(gate->data()),
                           reinterpret_cast<const llaisys::bf16_t *>(up->data()),
                           total_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops
