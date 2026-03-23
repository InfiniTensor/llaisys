#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>
#include <type_traits>

namespace {
	template <typename T>
	void argmax_impl(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel) {
		// Work in float for fp16/bf16 comparisons to avoid precision issues.
		using value_t = T;
		const value_t *v = reinterpret_cast<const value_t *>(vals);
		int64_t *out_idx = reinterpret_cast<int64_t *>(max_idx);
		value_t *out_val = reinterpret_cast<value_t *>(max_val);

		float best = llaisys::utils::cast<float>(v[0]);
		int64_t best_idx = 0;
		for (size_t i = 1; i < numel; ++i) {
			float cur = llaisys::utils::cast<float>(v[i]);
			if (cur > best) {
				best = cur;
				best_idx = static_cast<int64_t>(i);
			}
		}

		*out_idx = best_idx;
		*out_val = llaisys::utils::cast<value_t>(best);
	}
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return argmax_impl<float>(max_idx, max_val, vals, numel);
	case LLAISYS_DTYPE_BF16:
		return argmax_impl<llaisys::bf16_t>(max_idx, max_val, vals, numel);
	case LLAISYS_DTYPE_F16:
		return argmax_impl<llaisys::fp16_t>(max_idx, max_val, vals, numel);
	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}
} // namespace llaisys::ops::cpu
