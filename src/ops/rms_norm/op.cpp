#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include <cmath>

#include "../../utils.hpp"

#include <vector>

namespace {
template <typename T>
void rms_norm_impl(T *out_ptr, const T *in_ptr, const T *weight_ptr,
                   size_t outer_size, size_t norm_dim, float eps) {
    for (size_t i = 0; i < outer_size; ++i) {
        // compute rms，转换为float计算避免精度损失
        float rms = 0.0f;
        for (size_t j = 0; j < norm_dim; ++j) {
            float val = llaisys::utils::cast<float>(in_ptr[i * norm_dim + j]);
            rms += val * val;
        }
        rms = std::sqrt(rms / static_cast<float>(norm_dim) + eps);
        // normalize and scale
        for (size_t j = 0; j < norm_dim; ++j) {
            float val = llaisys::utils::cast<float>(in_ptr[i * norm_dim + j]);
            float weight_val = llaisys::utils::cast<float>(weight_ptr[j]);
            out_ptr[i * norm_dim + j] = llaisys::utils::cast<T>((val / rms) * weight_val);
        }
    }
}
} // namespace

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous.");
    CHECK_ARGUMENT(out->ndim() == in->ndim(), "RMSNorm: out and in must have the same number of dimensions.");
    for (size_t i = 0; i < out->ndim() - 1; ++i) {
        CHECK_ARGUMENT(out->shape()[i] == in->shape()[i],
                       "RMSNorm: out and in must have the same shape except for the last dimension.");
    }
    size_t norm_dim = out->shape().back();
    CHECK_ARGUMENT(weight->ndim() == 1 && weight->shape()[0] == norm_dim,
                   "RMSNorm: weight must be 1D with size equal to the last dimension of in/out.");
    // outer_size: number of RMSNorm operations to perform
    // norm_dim: size of each RMSNorm operation
    size_t outer_size = out->numel() / norm_dim;
    auto type = out->dtype();

    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        auto &ctx = llaisys::core::context();
        ctx.setDevice(out->deviceType(), out->deviceId());
        const auto *api = ctx.runtime().api();

        const size_t out_bytes = out->numel() * out->elementSize();
        const size_t in_bytes = in->numel() * in->elementSize();
        const size_t w_bytes = weight->numel() * weight->elementSize();

        std::vector<std::byte> h_out(out_bytes), h_in(in_bytes), h_w(w_bytes);
        api->memcpy_sync(h_in.data(), in->data(), in_bytes, LLAISYS_MEMCPY_D2H);
        api->memcpy_sync(h_w.data(), weight->data(), w_bytes, LLAISYS_MEMCPY_D2H);

        switch (type) {
        case LLAISYS_DTYPE_F32:
            rms_norm_impl(reinterpret_cast<float *>(h_out.data()),
                          reinterpret_cast<const float *>(h_in.data()),
                          reinterpret_cast<const float *>(h_w.data()),
                          outer_size, norm_dim, eps);
            break;
        case LLAISYS_DTYPE_F16:
            rms_norm_impl(reinterpret_cast<llaisys::fp16_t *>(h_out.data()),
                          reinterpret_cast<const llaisys::fp16_t *>(h_in.data()),
                          reinterpret_cast<const llaisys::fp16_t *>(h_w.data()),
                          outer_size, norm_dim, eps);
            break;
        case LLAISYS_DTYPE_BF16:
            rms_norm_impl(reinterpret_cast<llaisys::bf16_t *>(h_out.data()),
                          reinterpret_cast<const llaisys::bf16_t *>(h_in.data()),
                          reinterpret_cast<const llaisys::bf16_t *>(h_w.data()),
                          outer_size, norm_dim, eps);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }

        api->memcpy_sync(out->data(), h_out.data(), out_bytes, LLAISYS_MEMCPY_H2D);
        return;
    }

    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl(reinterpret_cast<float *>(out->data()),
                             reinterpret_cast<const float *>(in->data()),
                             reinterpret_cast<const float *>(weight->data()),
                             outer_size, norm_dim, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                             reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                             reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                             outer_size, norm_dim, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                             reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                             reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                             outer_size, norm_dim, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops
