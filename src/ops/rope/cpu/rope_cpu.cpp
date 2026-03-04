#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {
// 在 CPU 上实现 Rotary Position Embedding (RoPE)
// 输入张量形状：[seqlen, nhead, dim]
// pos_ids: 每个 token 的位置索引，长度为 seqlen
template <typename T>
void rope_impl(std::byte *output, const std::byte *input, const std::byte *position_ids,
               size_t sequence_length, size_t num_heads, size_t head_dim, float base_theta) {
    const T *input_data = reinterpret_cast<const T *>(input);
    const int64_t *pos_data = reinterpret_cast<const int64_t *>(position_ids);
    T *output_data = reinterpret_cast<T *>(output);

    const size_t head_stride = head_dim;
    const size_t seq_stride = num_heads * head_dim;
    const size_t half_dim = head_dim / 2; // RoPE 假设 dim 为偶数

    for (size_t seq_idx = 0; seq_idx < sequence_length; ++seq_idx) {
        const float position = static_cast<float>(pos_data[seq_idx]);

        for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            const T *x = input_data + seq_idx * seq_stride + head_idx * head_stride;
            T *y = output_data + seq_idx * seq_stride + head_idx * head_stride;

            for (size_t feat_idx = 0; feat_idx < half_dim; ++feat_idx) {
                // 计算旋转角度：θ_j = p / (θ^(2j/d))
                float freq_exponent = 2.0f * static_cast<float>(feat_idx) / static_cast<float>(head_dim);
                float angle = position / std::pow(base_theta, freq_exponent);
                float sin_val = std::sin(angle);
                float cos_val = std::cos(angle);

                // 分别读取前半部分 (a) 和后半部分 (b)
                float a = llaisys::utils::cast<float>(x[feat_idx]);
                float b = llaisys::utils::cast<float>(x[half_dim + feat_idx]);

                // 应用旋转：[a, b] → [a·cos - b·sin, b·cos + a·sin]
                y[feat_idx] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                y[half_dim + feat_idx] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
            }
        }
    }
}
}

namespace llaisys::ops::cpu {

// 调度 RoPE 操作到 CPU 后端
void rope(std::byte *output, const std::byte *input, const std::byte *pos_ids,
          llaisysDataType_t dtype, size_t seqlen, size_t nhead, size_t dim, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_impl<float>(output, input, pos_ids, seqlen, nhead, dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_impl<llaisys::bf16_t>(output, input, pos_ids, seqlen, nhead, dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_impl<llaisys::fp16_t>(output, input, pos_ids, seqlen, nhead, dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu