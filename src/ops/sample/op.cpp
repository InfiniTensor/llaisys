#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/sample_cpu.hpp"
#include <vector>
#include <cstring>

namespace llaisys::ops {

void sample(tensor_t next_token_id, tensor_t logits, float temperature, int top_k, float top_p) {
    ASSERT(next_token_id->dtype() == LLAISYS_DTYPE_I64 || next_token_id->dtype() == LLAISYS_DTYPE_I32, 
           "Sample: next_token_id must be integer type.");
    
    // 强制要求 logits 为 F32，防止在 CPU 端做复杂的半精度解析
    // (通常在 Python 端生成 logits 时已经做了 float() 转换)
    ASSERT(logits->dtype() == LLAISYS_DTYPE_F32, "Sample: logits must be F32.");

    size_t vocab_size = logits->numel();
    std::vector<float> cpu_logits(vocab_size);

    // --- 1. 使用框架自带的抽象 API 将 Logits 拷贝到 CPU，彻底摆脱 CUDA 依赖 ---
    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(cpu_logits.data(), logits->data(), vocab_size * sizeof(float));
    } else {
        // 切换到张量所在的设备上下文
        llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());
        // 使用通用的 memcpy_sync 接口 (Device To Host)
        llaisys::core::context().runtime().api()->memcpy_sync(
            cpu_logits.data(), 
            logits->data(), 
            vocab_size * sizeof(float), 
            LLAISYS_MEMCPY_D2H
        );
    }

    // --- 2. 执行 CPU 采样算法 ---
    int64_t sampled_id = 0;
    cpu::sample_f32(&sampled_id, cpu_logits.data(), vocab_size, temperature, top_k, top_p);

    // --- 3. 将结果写回 next_token_id ---
    if (next_token_id->deviceType() == LLAISYS_DEVICE_CPU) {
        if (next_token_id->dtype() == LLAISYS_DTYPE_I64) {
            reinterpret_cast<int64_t*>(next_token_id->data())[0] = sampled_id;
        } else if (next_token_id->dtype() == LLAISYS_DTYPE_I32) {
            reinterpret_cast<int32_t*>(next_token_id->data())[0] = static_cast<int32_t>(sampled_id);
        }
    } else {
        llaisys::core::context().setDevice(next_token_id->deviceType(), next_token_id->deviceId());
        if (next_token_id->dtype() == LLAISYS_DTYPE_I64) {
            llaisys::core::context().runtime().api()->memcpy_sync(
                next_token_id->data(), &sampled_id, sizeof(int64_t), LLAISYS_MEMCPY_H2D
            );
        } else {
            int32_t id32 = static_cast<int32_t>(sampled_id);
            llaisys::core::context().runtime().api()->memcpy_sync(
                next_token_id->data(), &id32, sizeof(int32_t), LLAISYS_MEMCPY_H2D
            );
        }
    }
}

} // namespace llaisys::ops