#include "op.hpp"
#include "../../utils.hpp"
#include <cmath> // 需要用到 std::sqrt

namespace llaisys::ops {

template <typename T>
void rms_norm_cpu(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 1. 获取维度
    // in: [M, N] (Batch/Seq, Hidden_Dim)
    // weight: [N]
    size_t M = in->shape()[0]; // 行数 (Sequence Length)
    size_t N = in->shape()[1]; // 列数 (Hidden Dimension)

    // 2. 获取指针
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    const T* w_ptr = reinterpret_cast<const T*>(weight->data());

    // 3. 逐行处理
    for (size_t i = 0; i < M; ++i) {
        float sum_sq = 0.0f;
        
        // 定位到当前行的起始位置
        const T* row_in = in_ptr + i * N;
        T* row_out = out_ptr + i * N;

        // --- 步骤 A: 计算平方和 ---
        for (size_t j = 0; j < N; ++j) {
            float val = utils::cast<float>(row_in[j]);
            sum_sq += val * val;
        }

        // --- 步骤 B: 计算缩放因子 (1 / RMS) ---
        float mean_sq = sum_sq / N;
        float rms = std::sqrt(mean_sq + eps);
        float scale = 1.0f / rms;

        // --- 步骤 C: 归一化并乘以权重 ---
        for (size_t j = 0; j < N; ++j) {
            float val = utils::cast<float>(row_in[j]);
            float w = utils::cast<float>(w_ptr[j]);
            
            // 公式: out = (in * scale) * weight
            float res = val * scale * w;
            
            row_out[j] = utils::cast<T>(res);
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:
            rms_norm_cpu<float>(out, in, weight, eps);
            break;
        case LLAISYS_DTYPE_F16:
            rms_norm_cpu<fp16_t>(out, in, weight, eps);
            break;
        case LLAISYS_DTYPE_BF16:
            rms_norm_cpu<bf16_t>(out, in, weight, eps);
            break;
        default:
            break;
    }
}

} // namespace llaisys::ops