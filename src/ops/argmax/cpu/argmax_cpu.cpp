#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>
#include <type_traits>

namespace {
    template <typename T>
    void argmax_impl(std::byte *index_out, std::byte *value_out, const std::byte *input_data, size_t element_count) {
        // 为避免半精度类型的精度损失，统一在 float 域中进行比较
        using value_type = T;
        const value_type *data_ptr = reinterpret_cast<const value_type *>(input_data);
        int64_t *result_index = reinterpret_cast<int64_t *>(index_out);
        value_type *result_value = reinterpret_cast<value_type *>(value_out);

        float current_max = llaisys::utils::cast<float>(data_ptr[0]);
        int64_t argmax_pos = 0;

        // 从第二个元素开始遍历（索引 1 到 end）
        for (size_t pos = 1; pos < element_count; ++pos) {
            float candidate = llaisys::utils::cast<float>(data_ptr[pos]);
            if (candidate > current_max) {
                current_max = candidate;
                argmax_pos = static_cast<int64_t>(pos);
            }
        }

        *result_index = argmax_pos;
        *result_value = llaisys::utils::cast<value_type>(current_max);
    }
}

namespace llaisys::ops::cpu {
    // 在 CPU 上执行 argmax 操作：返回最大值及其索引
    void argmax(std::byte *max_index, std::byte *max_value, const std::byte *values,
                llaisysDataType_t dtype, size_t num_elements) {
        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return argmax_impl<float>(max_index, max_value, values, num_elements);
        case LLAISYS_DTYPE_BF16:
            return argmax_impl<llaisys::bf16_t>(max_index, max_value, values, num_elements);
        case LLAISYS_DTYPE_F16:
            return argmax_impl<llaisys::fp16_t>(max_index, max_value, values, num_elements);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    }
} // namespace llaisys::ops::cpu