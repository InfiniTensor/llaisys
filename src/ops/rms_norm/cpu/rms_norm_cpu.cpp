#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {
	template <typename T>
	void rms_norm_impl(std::byte *out, const std::byte *in, const std::byte *weight, size_t rows, size_t cols,
	                  float eps) {
		const T *in_ptr = reinterpret_cast<const T *>(in);
		const T *w_ptr = reinterpret_cast<const T *>(weight);
		T *out_ptr = reinterpret_cast<T *>(out);

		for (size_t i = 0; i < rows; ++i) {
			const T *row_in = in_ptr + i * cols;
			T *row_out = out_ptr + i * cols;

			float sum_sq = 0.f;
			for (size_t j = 0; j < cols; ++j) {
				float v = llaisys::utils::cast<float>(row_in[j]);
				sum_sq += v * v;
			}
			float mean = sum_sq / static_cast<float>(cols);
			float inv_rms = 1.0f / std::sqrt(mean + eps);

			for (size_t j = 0; j < cols; ++j) {
				float v = llaisys::utils::cast<float>(row_in[j]);
				float w = llaisys::utils::cast<float>(w_ptr[j]);
				row_out[j] = llaisys::utils::cast<T>(v * inv_rms * w);
			}
		}
	}
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t rows, size_t cols, float eps) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return rms_norm_impl<float>(out, in, weight, rows, cols, eps);
	case LLAISYS_DTYPE_BF16:
		return rms_norm_impl<llaisys::bf16_t>(out, in, weight, rows, cols, eps);
	case LLAISYS_DTYPE_F16:
		return rms_norm_impl<llaisys::fp16_t>(out, in, weight, rows, cols, eps);
	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}
} // namespace llaisys::ops::cpu
