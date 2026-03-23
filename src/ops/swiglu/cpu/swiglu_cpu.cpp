#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {
	template <typename T>
	void swiglu_impl(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel) {
		const T *g_ptr = reinterpret_cast<const T *>(gate);
		const T *u_ptr = reinterpret_cast<const T *>(up);
		T *o_ptr = reinterpret_cast<T *>(out);

		for (size_t i = 0; i < numel; ++i) {
			float g = llaisys::utils::cast<float>(g_ptr[i]);
			float u = llaisys::utils::cast<float>(u_ptr[i]);
			float sigmoid = 1.0f / (1.0f + std::exp(-g));
			o_ptr[i] = llaisys::utils::cast<T>(u * g * sigmoid);
		}
	}
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return swiglu_impl<float>(out, gate, up, numel);
	case LLAISYS_DTYPE_BF16:
		return swiglu_impl<llaisys::bf16_t>(out, gate, up, numel);
	case LLAISYS_DTYPE_F16:
		return swiglu_impl<llaisys::fp16_t>(out, gate, up, numel);
	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}
} // namespace llaisys::ops::cpu
