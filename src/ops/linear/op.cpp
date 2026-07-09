#include "op.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {

template <typename T>
void linear_cpu(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 1. 获取矩阵维度
    // in: [M, K]
    // weight: [N, K]
    // out: [M, N]
    size_t M = in->shape()[0];      // Batch size 或 Sequence length
    size_t K = in->shape()[1];      // Input features
    size_t N = weight->shape()[0];  // Output features

    // 2. 获取数据指针
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    const T* w_ptr = reinterpret_cast<const T*>(weight->data());
    
    // 3. 处理 bias (它是可选的)
    // 检查 bias 是否存在且不为空
    const T* b_ptr = (bias && bias->numel() > 0) ? reinterpret_cast<const T*>(bias->data()) : nullptr;

    // 4. 核心循环：矩阵乘法 (Matrix Multiplication)
    // Y[m, n] = dot(X[m, :], W[n, :]) + b[n]
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            
            // 计算点积 (Dot Product)
            // 注意：由于 weight 是 [N, K]存储的，它的第 n 行就是连续的 K 个数
            // 这对 CPU 缓存非常友好，不需要手动转置
            for (size_t k = 0; k < K; ++k) {
                float x_val = utils::cast<float>(in_ptr[m * K + k]);
                float w_val = utils::cast<float>(w_ptr[n * K + k]);
                sum += x_val * w_val;
            }
            
            // 加上 bias
            if (b_ptr) {
                sum += utils::cast<float>(b_ptr[n]);
            }
            
            // 写入结果
            out_ptr[m * N + n] = utils::cast<T>(sum);
        }
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:
            linear_cpu<float>(out, in, weight, bias);
            break;
        case LLAISYS_DTYPE_F16:
            linear_cpu<fp16_t>(out, in, weight, bias);
            break;
        case LLAISYS_DTYPE_BF16:
            linear_cpu<bf16_t>(out, in, weight, bias);
            break;
        default:
            break;
    }
}

} // namespace llaisys::ops