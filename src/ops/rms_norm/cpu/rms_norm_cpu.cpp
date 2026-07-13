#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

/**
 * RMS Norm 核心计算
 *
 * 内存布局 (行优先):
 *   input/out 视为 [batch, hidden_size] 的二维数组
 *   weight 是长度为 hidden_size 的一维数组
 *
 * 计算流程 (对每个 batch i):
 *   1. 计算 RMS_i = sqrt(mean(input[i][:]^2) + eps)
 *   2. out[i][j] = (input[i][j] / RMS_i) * weight[j]
 *
 * 索引计算:
 *   - input[i][j] = input[i * hidden_size + j]
 *   - weight[j] 对所有 batch 共享
 */
template <typename T>
void rms_norm_(T *out, const T *input, const T *weight, size_t batch, size_t hidden_size, float eps) {
    // 遍历每个独立的向量 (batch 维度)
    for (size_t i = 0; i < batch; i++) {
        // Step 1: 计算 sum(x^2)，在 hidden_size 维度上累加
        float sum_sq = 0.0F;
        for (size_t j = 0; j < hidden_size; j++) {
            float val = llaisys::utils::cast<float>(input[i * hidden_size + j]);
            sum_sq += val * val;
        }
        // Step 2: 计算 RMS = sqrt(mean(x^2) + eps)
        float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);

        // Step 3: 归一化并乘以 weight
        for (size_t j = 0; j < hidden_size; j++) {
            float val = llaisys::utils::cast<float>(input[i * hidden_size + j]);
            float wgt = llaisys::utils::cast<float>(weight[j]);
            out[i * hidden_size + j] = llaisys::utils::cast<T>((val / rms) * wgt);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *input, const std::byte *weight, llaisysDataType_t type, size_t batch, size_t hidden_size, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(input),
                         reinterpret_cast<const float *>(weight), batch, hidden_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(input),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), batch, hidden_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(input),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), batch, hidden_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
