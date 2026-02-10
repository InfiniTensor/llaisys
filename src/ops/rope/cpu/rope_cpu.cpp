#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seq_len, size_t num_heads, size_t head_dim) {
    
    size_t half_dim = head_dim / 2;
    
    std::vector<float> denoms(half_dim);
    for (size_t j = 0; j < half_dim; ++j) {
        // 此处用 double 以防止 pow 计算时出现精度问题
        double exponent = (2.0 * static_cast<double>(j)) / static_cast<double>(head_dim);
        double denom_d = std::pow(static_cast<double>(theta), exponent);
        denoms[j] = static_cast<float>(denom_d);
    }

    for (size_t s = 0; s < seq_len; ++s) {
        int64_t pos = pos_ids[s];
        float pos_f = static_cast<float>(pos);

        for (size_t h = 0; h < num_heads; ++h) {
            size_t offset = s * (num_heads * head_dim) + h * head_dim;

            const T* src_vec = in + offset;
            T* dst_vec = out + offset;

            for (size_t j = 0; j < half_dim; ++j) {
                float angle = pos_f / denoms[j];

                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);

                float a = llaisys::utils::cast<float>(src_vec[j]);
                float b = llaisys::utils::cast<float>(src_vec[j + half_dim]);

                float a_out = a * cos_val - b * sin_val;
                float b_out = b * cos_val + a * sin_val;

                dst_vec[j] = llaisys::utils::cast<T>(a_out);
                dst_vec[j + half_dim] = llaisys::utils::cast<T>(b_out);
            }
        }
    }
}

namespace llaisys::ops::cpu {

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t dtype, const std::vector<size_t> &shape) {
    
    size_t seq_len = shape[0];
    size_t num_heads = shape[1];
    size_t head_dim = shape[2];

    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos_ptr, theta, seq_len, num_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), pos_ptr, theta, seq_len, num_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), pos_ptr, theta, seq_len, num_heads, head_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu