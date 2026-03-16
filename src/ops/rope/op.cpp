#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <vector>
#ifdef ENABLE_NVIDIA_API
#include "llaisys/ops_nvidia.h"
#endif

namespace {

// RoPE: phi_{i,j} = p_i / theta^(2j/d); a'_j = a_j*cos(phi) - b_j*sin(phi), b'_j = b_j*cos(phi) + a_j*sin(phi)
// in/out layout: [seq_len, n_head, head_dim], 连续
template <typename T>
void rope_impl(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t n_head, size_t head_dim,
              float theta) {
    const size_t half_d = head_dim / 2;
    std::vector<double> inv_freq(static_cast<size_t>(half_d));
    for (size_t j = 0; j < half_d; j++) {
        double exp = 2.0 * static_cast<double>(j) / static_cast<double>(head_dim);
        inv_freq[j] = 1.0 / std::pow(static_cast<double>(theta), exp);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t s = 0; s < seq_len; s++) {
        double p = static_cast<double>(pos_ids[s]);
        for (size_t h = 0; h < n_head; h++) {
            const T *row = in + s * n_head * head_dim + h * head_dim;
            T *row_out = out + s * n_head * head_dim + h * head_dim;
            for (size_t j = 0; j < half_d; j++) {
                double phi = p * inv_freq[j];
                double cos_phi = std::cos(phi);
                double sin_phi = std::sin(phi);

                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(row[j]);
                    b = llaisys::utils::cast<float>(row[half_d + j]);
                } else {
                    a = static_cast<float>(row[j]);
                    b = static_cast<float>(row[half_d + j]);
                }
                float a_out = static_cast<float>(a * cos_phi - b * sin_phi);
                float b_out = static_cast<float>(b * cos_phi + a * sin_phi);
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    row_out[j] = llaisys::utils::cast<T>(a_out);
                    row_out[half_d + j] = llaisys::utils::cast<T>(b_out);
                } else {
                    row_out[j] = static_cast<T>(a_out);
                    row_out[half_d + j] = static_cast<T>(b_out);
                }
            }
        }
    }
}

void rope_cpu(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t dtype,
              size_t seq_len, size_t n_head, size_t head_dim, float theta) {
    const int64_t *pid = reinterpret_cast<const int64_t *>(pos_ids);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_impl(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pid, seq_len, n_head,
                  head_dim, theta);
        return;
    case LLAISYS_DTYPE_F16:
        rope_impl(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), pid,
                  seq_len, n_head, head_dim, theta);
        return;
    case LLAISYS_DTYPE_BF16:
        rope_impl(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), pid,
                  seq_len, n_head, head_dim, theta);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DEVICE(out, pos_ids);
    ASSERT(out->dtype() == in->dtype(), "rope: out and in must have same dtype");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be int64");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: out, in, pos_ids must be contiguous");
    ASSERT(out->ndim() == 3 && in->ndim() == 3, "rope: out and in must be 3D [seqlen, nhead, d]");
    ASSERT(pos_ids->ndim() == 1, "rope: pos_ids must be 1D [seqlen]");
    size_t seq_len = in->shape()[0], n_head = in->shape()[1], head_dim = in->shape()[2];
    ASSERT(head_dim % 2 == 0, "rope: head_dim must be even");
    ASSERT(out->shape()[0] == seq_len && out->shape()[1] == n_head && out->shape()[2] == head_dim,
           "rope: out shape must match in");
    ASSERT(pos_ids->numel() == seq_len, "rope: pos_ids length must equal seq_len");

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return rope_cpu(out->data(), in->data(), pos_ids->data(), out->dtype(), seq_len, n_head, head_dim,
                        theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::rope(out->data(), in->data(), reinterpret_cast<const int64_t *>(pos_ids->data()), out->dtype(), seq_len, n_head, head_dim, theta);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
