#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seq_length, size_t head_nums, size_t head_dim, float theta) {
    const size_t max_i = head_dim / 2;
    for (size_t s = 0; s < seq_length; s++) {
        for (size_t h = 0; h < head_nums; h++) {
            for (size_t i = 0; i < max_i; i++) {
				float angle = pos_ids[s] / std::pow(theta, (2.0f * i) / head_dim);
				float cos_angle = std::cos(angle);
				float sin_angle = std::sin(angle);

				size_t base = s * head_nums * head_dim + h * head_dim;
				size_t index_a = base + i;
				size_t index_b = base + i + max_i;
				if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
					float x1 = llaisys::utils::cast<float>(in[index_a]);
					float x2 = llaisys::utils::cast<float>(in[index_b]);
					out[index_a] = llaisys::utils::cast<T>(x1 * cos_angle - x2 * sin_angle);
					out[index_b] = llaisys::utils::cast<T>(x2 * cos_angle + x1 * sin_angle);
				} else {
					float x1 = in[index_a];
					float x2 = in[index_b];
					out[index_a] = x1 * cos_angle - x2 * sin_angle;
					out[index_b] = x2 * cos_angle + x1 * sin_angle;
				}
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seq_length, size_t head_nums, size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), seq_length, head_nums, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), seq_length, head_nums, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), seq_length, head_nums, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
