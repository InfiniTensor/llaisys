#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {
	template <typename T>
	void rope_impl(std::byte *out, const std::byte *in, const std::byte *pos_ids,
	              size_t seqlen, size_t nhead, size_t dim, float theta) {
		const T *in_ptr = reinterpret_cast<const T *>(in);
		const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);
		T *out_ptr = reinterpret_cast<T *>(out);

		size_t head_stride = dim;
		size_t seq_stride = nhead * dim;
		size_t half = dim / 2;

		for (size_t s = 0; s < seqlen; ++s) {
			float p = static_cast<float>(pos_ptr[s]);
			for (size_t h = 0; h < nhead; ++h) {
				const T *x = in_ptr + s * seq_stride + h * head_stride;
				T *y = out_ptr + s * seq_stride + h * head_stride;

				for (size_t j = 0; j < half; ++j) {
					float exponent = static_cast<float>(2.0f * static_cast<float>(j) / static_cast<float>(dim));
					float angle = p / std::pow(theta, exponent);
					float sinv = std::sin(angle);
					float cosv = std::cos(angle);

					float a = llaisys::utils::cast<float>(x[j]);
					float b = llaisys::utils::cast<float>(x[half + j]);

					y[j] = llaisys::utils::cast<T>(a * cosv - b * sinv);
					y[half + j] = llaisys::utils::cast<T>(b * cosv + a * sinv);
				}
			}
		}
	}
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type,
          size_t seqlen, size_t nhead, size_t dim, float theta) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return rope_impl<float>(out, in, pos_ids, seqlen, nhead, dim, theta);
	case LLAISYS_DTYPE_BF16:
		return rope_impl<llaisys::bf16_t>(out, in, pos_ids, seqlen, nhead, dim, theta);
	case LLAISYS_DTYPE_F16:
		return rope_impl<llaisys::fp16_t>(out, in, pos_ids, seqlen, nhead, dim, theta);
	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}
} // namespace llaisys::ops::cpu
