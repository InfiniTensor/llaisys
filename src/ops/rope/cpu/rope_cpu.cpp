#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                 size_t seqlen, size_t num_heads, size_t d, float theta) {
    size_t half_d = d / 2;  // j=0,1,…,d/2−1。
    
    // 优化：预计算 inv_freq 以避免在循环中重复调用昂贵的 std::pow
    std::vector<double> inv_freq_vec(half_d);
    for (size_t j = 0; j < half_d; ++j) {
        // RoPE的角度频率计算公式 1/sita^(2j/d)
        inv_freq_vec[j] = 1.0 / std::pow(static_cast<double>(theta), static_cast<double>(2 * j) / static_cast<double>(d));
    }

    // 优化：预计算当前所有位置的 cos 和 sin 值
    // 因为 cos/sin 只与位置 i 和维度 j 有关，与 head 无关
    std::vector<float> cos_table(seqlen * half_d);
    std::vector<float> sin_table(seqlen * half_d);

    #pragma omp parallel for
    for (size_t i = 0; i < seqlen; ++i) { 
        double pos = static_cast<double>(pos_ids[i]);
        for (size_t j = 0; j < half_d; ++j) { // 
            // 因为所有 num_heads 个头在相同位置使用的旋转角度是一模一样的，所以我们只算一份，存入这张表，让所有头共享
            double phi = pos * inv_freq_vec[j];
            cos_table[i * half_d + j] = static_cast<float>(std::cos(phi));
            sin_table[i * half_d + j] = static_cast<float>(std::sin(phi));
        }
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seqlen; ++i) { // i-表示第i个token
        for (size_t h = 0; h < num_heads; ++h) { // h-表示第h个head
    //  定位当前 Token 的当前 Head 的内存起始位置,每个Token包含num_heads个头，每个头有d个元素。
    //  所以一个Token占用的总元素个数=num_heads*d , 每个 Head 占用 d 个元素。  
    //  head_in和head_out是in和out的子数组，指向第i个token的第h个头起始位置      
            const T *head_in = in + (i * num_heads * d) + (h * d);
            T *head_out = out + (i * num_heads * d) + (h * d);
            
            for (size_t j = 0; j < half_d; ++j) {
                // 使用预计算的三角函数值，避免在 head 循环中重复计算
                float cos_phi = cos_table[i * half_d + j];
                float sin_phi = sin_table[i * half_d + j];

                // 提取 a 和 b 的值 (a 在前半部分，b 在后半部分)
    // 代码采用的方式将向量平分成左右两半，左半部分的第j个元素与右半部分的第j个元素配对
                float a = llaisys::utils::cast<float>(head_in[j]);
                float b = llaisys::utils::cast<float>(head_in[j + half_d]);

                // 3. 应用旋转矩阵公式：
                // [a']   [cos  -sin] [a] 
                // [b'] = [sin   cos] [b] 
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