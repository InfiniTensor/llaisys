#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                 size_t seqlen, size_t num_heads, size_t d, float theta) {
    size_t half_d = d / 2;
    
    // 优化：预计算 inv_freq 以避免在循环中重复调用昂贵的 std::pow
    std::vector<float> inv_freq_vec(half_d);
    for (size_t j = 0; j < half_d; ++j) {
        // RoPE的角度频率计算公式 1/sita^(2j/d)
        inv_freq_vec[j] = 1.0f / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(d));
    }

    // 优化：预计算当前所有位置的 cos 和 sin 值
    // 因为 cos/sin 只与位置 i 和维度 j 有关，与 head 无关
    std::vector<float> cos_table(seqlen * half_d);
    std::vector<float> sin_table(seqlen * half_d);
    #pragma omp parallel for
    for (size_t i = 0; i < seqlen; ++i) {
        float pos = static_cast<float>(pos_ids[i]);
        for (size_t j = 0; j < half_d; ++j) {
            float phi = pos * inv_freq_vec[j];
            cos_table[i * half_d + j] = std::cos(phi);
            sin_table[i * half_d + j] = std::sin(phi);
        }
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t h = 0; h < num_heads; ++h) {
            // 定位当前 head 的起始位置
            const T *head_in = in + (i * num_heads * d) + (h * d);
            T *head_out = out + (i * num_heads * d) + (h * d);
            
            for (size_t j = 0; j < half_d; ++j) {
                // 使用预计算的三角函数值，避免在 head 循环中重复计算
                float cos_phi = cos_table[i * half_d + j];
                float sin_phi = sin_table[i * half_d + j];

                // 提取 a 和 b 的值 (a 在前半部分，b 在后半部分)
                float a = llaisys::utils::cast<float>(head_in[j]);
                float b = llaisys::utils::cast<float>(head_in[j + half_d]);

                // 应用旋转公式
                float a_prime = a * cos_phi - b * sin_phi;
                float b_prime = b * cos_phi + a * sin_phi;

                // 写回结果
                head_out[j] = llaisys::utils::cast<T>(a_prime);
                head_out[j + half_d] = llaisys::utils::cast<T>(b_prime);
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t dtype, size_t seqlen, size_t num_heads, size_t d, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float>((float *)out, (const float *)in, pos_ids, seqlen, num_heads, d, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<llaisys::fp16_t>((llaisys::fp16_t *)out, (const llaisys::fp16_t *)in, pos_ids, seqlen, num_heads, d, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<llaisys::bf16_t>((llaisys::bf16_t *)out, (const llaisys::bf16_t *)in, pos_ids, seqlen, num_heads, d, theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu