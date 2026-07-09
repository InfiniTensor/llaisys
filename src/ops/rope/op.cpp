#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys::ops {

template <typename T>
void rope_cpu(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t head_dim = in->shape()[2];
    size_t half_dim = head_dim / 2;

    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());

    for (size_t s = 0; s < seqlen; ++s) {
        int64_t pos = pos_ptr[s];
        
        for (size_t h = 0; h < nhead; ++h) {
            size_t offset = (s * nhead + h) * head_dim;

            for (size_t j = 0; j < half_dim; ++j) {
                // --- 修改点：使用 double 进行高精度计算 ---
                // 这里的 theta 和 j 计算如果用 float 会有累积误差
                double freq = std::pow((double)theta, -2.0 * (double)j / (double)head_dim);
                double angle = (double)pos * freq;
                
                double cos_val = std::cos(angle);
                double sin_val = std::sin(angle);

                // 读取时转为 float 即可 (输入本身精度有限)
                float a = utils::cast<float>(in_ptr[offset + j]);
                float b = utils::cast<float>(in_ptr[offset + j + half_dim]);

                // 运算使用高精度
                float a_out = (float)(a * cos_val - b * sin_val);
                float b_out = (float)(b * cos_val + a * sin_val);

                out_ptr[offset + j] = utils::cast<T>(a_out);
                out_ptr[offset + j + half_dim] = utils::cast<T>(b_out);
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:
            rope_cpu<float>(out, in, pos_ids, theta);
            break;
        case LLAISYS_DTYPE_F16:
            rope_cpu<fp16_t>(out, in, pos_ids, theta);
            break;
        case LLAISYS_DTYPE_BF16:
            rope_cpu<bf16_t>(out, in, pos_ids, theta);
            break;
        default:
            break;
    }
}

} // namespace llaisys::ops