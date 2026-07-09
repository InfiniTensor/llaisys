#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include <cmath>

#include <vector>

namespace {
template <typename T>
void rope_impl(T *out_ptr, const T *in_ptr, const int64_t *pos_ids,
               size_t batch_size, size_t seq_len, size_t dim, float theta) {
    size_t half_dim = dim / 2;

    for (size_t s = 0; s < batch_size; ++s) {
        // 获取当前位置的ID
        int64_t pos_id = pos_ids[s];
        for (size_t h = 0; h < seq_len; ++h) {
            // 维度对循环
            // s是第几个token
            // h是第几个head
            // 每个token对应seq_len个head，列如hidden=512, num_head=8,则seq_len=64，实际上每个词对应64个head
            // 每个(s, h)对应一个长度为dim的向量，base就是这个向量的起始索引，dim是每个头的向量长度
            size_t base = (s * seq_len + h) * dim;
            for (size_t d = 0; d < half_dim; ++d) {
                const double exponent = (2.0 * static_cast<double>(d)) / static_cast<double>(dim);
                const double angle = static_cast<double>(pos_id) / std::pow(static_cast<double>(theta), exponent);
                const double cos_angle = std::cos(angle);
                const double sin_angle = std::sin(angle);
                // original values (split half/half)
                const double x1 = static_cast<double>(llaisys::utils::cast<float>(in_ptr[base + d]));
                const double x2 = static_cast<double>(llaisys::utils::cast<float>(in_ptr[base + half_dim + d]));
                // apply rotation
                out_ptr[base + d] = llaisys::utils::cast<T>(x1 * cos_angle - x2 * sin_angle);
                out_ptr[base + half_dim + d] = llaisys::utils::cast<T>(x2 * cos_angle + x1 * sin_angle);
            }
        }
    }
}
} // namespace

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 检查设备和数据类型
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // 检查张量是否是连续存储的
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "Rope: all tensors must be contiguous.");

    // 检查维度和pos_ids数据类型
    CHECK_ARGUMENT(in->ndim() == 3, "Rope: in must be 3D.");
    CHECK_ARGUMENT(out->ndim() == in->ndim(), "Rope: out and in must have the same number of dimensions.");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Rope: pos_ids must be of dtype int64.");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "Rope: pos_ids must be 1D.");

    // 检查维度匹配
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "Rope: out.shape[0] must equal in.shape[0].");
    CHECK_ARGUMENT(out->shape()[1] == in->shape()[1], "Rope: out.shape[1] must equal in.shape[1].");
    CHECK_ARGUMENT(out->shape()[2] == in->shape()[2], "Rope: out.shape[2] must equal in.shape[2].");
    CHECK_ARGUMENT(in->shape()[0] == pos_ids->shape()[0], "Rope: in.shape[0] must equal pos_ids.shape[0].");

    // 维度必须为偶数
    size_t dim = in->shape()[2];
    CHECK_ARGUMENT(dim % 2 == 0, "Rope: the last dimension must be even.");

    // 获取数据类型用于分发
    auto type = in->dtype();

    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        auto &ctx = llaisys::core::context();
        ctx.setDevice(out->deviceType(), out->deviceId());
        const auto *api = ctx.runtime().api();

        const size_t p_bytes = pos_ids->numel() * pos_ids->elementSize();
        std::vector<int64_t> h_pos(pos_ids->numel());
        api->memcpy_sync(h_pos.data(), pos_ids->data(), p_bytes, LLAISYS_MEMCPY_D2H);

        switch (type) {
        case LLAISYS_DTYPE_F32: {
            std::vector<float> h_in(in->numel()), h_out(out->numel());
            api->memcpy_sync(h_in.data(), in->data(), in->numel() * sizeof(float), LLAISYS_MEMCPY_D2H);
            rope_impl(h_out.data(), h_in.data(), h_pos.data(), in->shape()[0], in->shape()[1], dim, theta);
            api->memcpy_sync(out->data(), h_out.data(), out->numel() * sizeof(float), LLAISYS_MEMCPY_H2D);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            std::vector<llaisys::fp16_t> h_in(in->numel()), h_out(out->numel());
            api->memcpy_sync(h_in.data(), in->data(), in->numel() * sizeof(llaisys::fp16_t), LLAISYS_MEMCPY_D2H);
            rope_impl(h_out.data(), h_in.data(), h_pos.data(), in->shape()[0], in->shape()[1], dim, theta);
            api->memcpy_sync(out->data(), h_out.data(), out->numel() * sizeof(llaisys::fp16_t), LLAISYS_MEMCPY_H2D);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            std::vector<llaisys::bf16_t> h_in(in->numel()), h_out(out->numel());
            api->memcpy_sync(h_in.data(), in->data(), in->numel() * sizeof(llaisys::bf16_t), LLAISYS_MEMCPY_D2H);
            rope_impl(h_out.data(), h_in.data(), h_pos.data(), in->shape()[0], in->shape()[1], dim, theta);
            api->memcpy_sync(out->data(), h_out.data(), out->numel() * sizeof(llaisys::bf16_t), LLAISYS_MEMCPY_H2D);
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }

        return;
    }

    // 检查设备类型
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    // 获取数据指针并调用相应的数据类型处理函数
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_impl(reinterpret_cast<float *>(out->data()),
                         reinterpret_cast<const float *>(in->data()),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         in->shape()[0], in->shape()[1], dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_impl(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                         reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         in->shape()[0], in->shape()[1], dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_impl(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                         reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         in->shape()[0], in->shape()[1], dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops
