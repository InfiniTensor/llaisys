#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath> // sin, cos, pow
#include <omp.h>

template <typename T>
void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                 size_t seq_len, size_t n_heads, size_t head_dim, float theta) {

    size_t half_dim = head_dim / 2;

// 遍历每一个 Token
#pragma omp parallel for
    for (size_t s = 0; s < seq_len; s++) {
        int64_t pos = pos_ids[s];

        // 遍历每一个 Head
        for (size_t h = 0; h < n_heads; h++) {

            size_t offset = s * (n_heads * head_dim) + h * head_dim;
            const T *src_vec = in + offset;
            T *dst_vec = out + offset;

            // 遍历每一对 (j)
            for (size_t j = 0; j < half_dim; j++) {

                // --- 1. 精度提升关键点 ---
                // 使用 double (64位) 进行中间计算，对齐 PyTorch 的精度行为
                double j_d = static_cast<double>(j);
                double d_d = static_cast<double>(head_dim);
                double theta_d = static_cast<double>(theta);
                double pos_d = static_cast<double>(pos);

                // 计算频率 (High Precision)
                double freq = std::pow(theta_d, -2.0 * j_d / d_d);

                // 计算角度 (High Precision)
                double angle = pos_d * freq;

                // 计算 Sin/Cos (High Precision)
                double cos_val = std::cos(angle);
                double sin_val = std::sin(angle);

                // --- 2. 读取数据并计算 ---
                // 这里也先转成 double 算完再转回去，减少误差累积
                double a = static_cast<double>(llaisys::utils::cast<float>(src_vec[j]));
                double b = static_cast<double>(llaisys::utils::cast<float>(src_vec[j + half_dim]));

                double a_out = a * cos_val - b * sin_val;
                double b_out = b * cos_val + a * sin_val;

                // --- 3. 存回结果 ---
                // 最后一步才转回目标类型 T (float/fp16/bf16)
                dst_vec[j] = llaisys::utils::cast<T>(static_cast<float>(a_out));
                dst_vec[j + half_dim] = llaisys::utils::cast<T>(static_cast<float>(b_out));
            }
        }
    }
}

namespace llaisys::ops::cpu {

void rope(const std::byte *out_data, const std::byte *in_data, const std::byte *pos_ids,
          llaisysDataType_t dtype,
          size_t seq_len, size_t n_heads, size_t head_dim, float theta) {

    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_kernel<float>(
            reinterpret_cast<float *>(const_cast<std::byte *>(out_data)),
            reinterpret_cast<const float *>(in_data),
            pos_ptr, seq_len, n_heads, head_dim, theta);

    case LLAISYS_DTYPE_F16:
        return rope_kernel<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(const_cast<std::byte *>(out_data)),
            reinterpret_cast<const llaisys::fp16_t *>(in_data),
            pos_ptr, seq_len, n_heads, head_dim, theta);

    case LLAISYS_DTYPE_BF16:
        return rope_kernel<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(const_cast<std::byte *>(out_data)),
            reinterpret_cast<const llaisys::bf16_t *>(in_data),
            pos_ptr, seq_len, n_heads, head_dim, theta);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu