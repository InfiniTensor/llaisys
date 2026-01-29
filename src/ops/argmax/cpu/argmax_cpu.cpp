#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include "../../../utils/types.hpp"
#include "../../../utils/check.hpp"
#include <limits>


namespace llaisys::ops::cpu {

template <typename T, typename IndexT>

// 它的核心任务是在一段连续的内存中找到最大值及其对应的索引。由于深度学习框架需要支持多种数据类型
// （如 Float32, Float16, BFloat16）和索引类型（Int32, Int64）
void argmax_kernel(const T *vals, T *max_val, IndexT *max_idx, size_t numel) {
    if (numel == 0) return;

    // 初始化：假设第一个元素是最大的
    size_t best_idx = 0;
    // 使用 cast<float> 确保 fp16/bf16 能正确转为 float 进行比较
    // CPU 通常没有原生的比较指令，或者它们被封装在结构体中。这里统一将它们转换为 float (32位浮点) 进行比较，既保证了精度，也简化了逻辑
    float max_v = utils::cast<float>(vals[0]);

    for (size_t i = 1; i < numel; ++i) {
        float curr_v = utils::cast<float>(vals[i]);
        if (curr_v > max_v) {
            max_v = curr_v;
            best_idx = i;
        }
    }

    // 将结果写回输出内存
    *max_val = utils::cast<T>(max_v);
    *max_idx = static_cast<IndexT>(best_idx);
}

// 辅助宏：用于处理不同的索引类型 (int32 或 int64)
#define DISPATCH_INDEX_TYPE(VAL_TYPE, VAL_T)                                   \
    switch (idx_dtype) {                                                       \
    case LLAISYS_DTYPE_I32:                                                    \
        argmax_kernel<VAL_T, int32_t>(                                         \
            reinterpret_cast<const VAL_T *>(vals),                             \
            reinterpret_cast<VAL_T *>(max_val),                                \
            reinterpret_cast<int32_t *>(max_idx),                              \
            numel);                                                            \
        break;                                                                 \
    case LLAISYS_DTYPE_I64:                                                    \
        argmax_kernel<VAL_T, int64_t>(                                         \
            reinterpret_cast<const VAL_T *>(vals),                             \
            reinterpret_cast<VAL_T *>(max_val),                                \
            reinterpret_cast<int64_t *>(max_idx),                              \
            numel);                                                            \
        break;                                                                 \
    default:                                                                   \
        EXCEPTION_UNSUPPORTED_DATATYPE(idx_dtype);                             \
    }

void argmax(const std::byte *vals, std::byte *max_val, std::byte *max_idx,
            llaisysDataType_t val_dtype, llaisysDataType_t idx_dtype, size_t numel) {
    
    // 第一层分发：根据数值类型 (vals/max_val)
    switch (val_dtype) {
    case LLAISYS_DTYPE_F32:
        DISPATCH_INDEX_TYPE(LLAISYS_DTYPE_F32, float);
        break;
    case LLAISYS_DTYPE_F16:
        DISPATCH_INDEX_TYPE(LLAISYS_DTYPE_F16, llaisys::fp16_t);
        break;
    case LLAISYS_DTYPE_BF16:
        DISPATCH_INDEX_TYPE(LLAISYS_DTYPE_BF16, llaisys::bf16_t);
        break;
    // 如果需要支持 int8/int32 等作为值类型，可在此添加 case
    case LLAISYS_DTYPE_I32:
        DISPATCH_INDEX_TYPE(LLAISYS_DTYPE_I32, int32_t);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_dtype);
    }
}

#undef DISPATCH_INDEX_TYPE

} // namespace llaisys::ops::cpu