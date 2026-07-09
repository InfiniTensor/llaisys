#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <vector>

namespace {
template <typename T>
void rearrange_impl(T *out_ptr, const T *in_ptr, const std::vector<size_t> &shape,
                    const std::vector<ptrdiff_t> &in_strides, size_t ndim) {
    // 多维递归遍历，处理任意形状的张量
    // out_ptr 写入位置始终连续（步长 = 后续所有维度的乘积）
    // in_ptr 读取位置按照 in_strides 跳跃

    if (ndim == 0) {
        return;
    }

    // 栈用于迭代而非递归（避免栈溢出）
    std::vector<size_t> indices(ndim, 0);

    // 计算输出的连续步长（每维后续维度的乘积）
    std::vector<size_t> out_strides(ndim);
    out_strides[ndim - 1] = 1;
    for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * shape[i + 1];
    }

    // 总元素数
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= shape[i];
    }

    // 遍历所有元素
    for (size_t elem_idx = 0; elem_idx < total_elements; ++elem_idx) {
        // 从线性索引计算多维索引
        size_t temp = elem_idx;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            indices[i] = temp % shape[i];
            temp /= shape[i];
        }

        // 根据 indices 计算两个指针的偏移
        ptrdiff_t in_offset = 0;
        size_t out_offset = 0;
        for (size_t i = 0; i < ndim; ++i) {
            in_offset += static_cast<ptrdiff_t>(indices[i]) * in_strides[i];
            out_offset += indices[i] * out_strides[i];
        }

        // 复制单个元素
        out_ptr[out_offset] = in_ptr[in_offset];
    }
}
} // namespace

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    // 基本验证
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // 形状检查：必须相同
    CHECK_ARGUMENT(out->ndim() == in->ndim(),
                   "rearrange: out and in must have the same number of dimensions.");
    CHECK_ARGUMENT(out->shape() == in->shape(),
                   "rearrange: out and in must have the same shape.");

    size_t ndim = out->ndim();
    auto type = out->dtype();

    // 获取步长信息
    const auto &in_strides = in->strides();
    const auto &shape = out->shape();

    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        auto &ctx = llaisys::core::context();
        ctx.setDevice(out->deviceType(), out->deviceId());
        const auto *api = ctx.runtime().api();

        const size_t out_bytes = out->numel() * out->elementSize();
        const size_t in_bytes = in->numel() * in->elementSize();
        std::vector<std::byte> h_out(out_bytes), h_in(in_bytes);
        api->memcpy_sync(h_in.data(), in->data(), in_bytes, LLAISYS_MEMCPY_D2H);

        switch (type) {
        case LLAISYS_DTYPE_F32:
            rearrange_impl(reinterpret_cast<float *>(h_out.data()),
                           reinterpret_cast<const float *>(h_in.data()),
                           shape, in_strides, ndim);
            break;
        case LLAISYS_DTYPE_F16:
            rearrange_impl(reinterpret_cast<llaisys::fp16_t *>(h_out.data()),
                           reinterpret_cast<const llaisys::fp16_t *>(h_in.data()),
                           shape, in_strides, ndim);
            break;
        case LLAISYS_DTYPE_BF16:
            rearrange_impl(reinterpret_cast<llaisys::bf16_t *>(h_out.data()),
                           reinterpret_cast<const llaisys::bf16_t *>(h_in.data()),
                           shape, in_strides, ndim);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }

        api->memcpy_sync(out->data(), h_out.data(), out_bytes, LLAISYS_MEMCPY_H2D);
        return;
    }

    // 设备检查
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    // dtype 分发
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_impl(reinterpret_cast<float *>(out->data()),
                              reinterpret_cast<const float *>(in->data()),
                              shape, in_strides, ndim);
    case LLAISYS_DTYPE_F16:
        return rearrange_impl(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                              reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                              shape, in_strides, ndim);
    case LLAISYS_DTYPE_BF16:
        return rearrange_impl(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                              reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                              shape, in_strides, ndim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops
