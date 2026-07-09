#include "op.hpp"
#include "../../utils.hpp" // 注意：如果报错找不到头文件，尝试改为 "../utils.hpp"
#include <limits>
#include <iostream>

namespace llaisys::ops {

// 定义一个模板函数，这样可以同时处理 float, fp16, bf16
template <typename T>
void argmax_cpu(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 1. 获取元素总数
    size_t count = vals->numel();
    
    // 2. 获取数据指针 (生肉 -> 熟肉)
    const T* vals_data = reinterpret_cast<const T*>(vals->data());
    
    // 3. 初始化：最大值设为负无穷，索引设为 0
    float max_v = -std::numeric_limits<float>::infinity();
    int64_t max_i = 0;

    // 4. 核心循环：打擂台找最大值
    for (size_t i = 0; i < count; i++) {
        // 关键点：统一转成 float 进行比较，避免半精度误差
        float val = utils::cast<float>(vals_data[i]);
        if (val > max_v) {
            max_v = val;
            max_i = i;
        }
    }

    // 5. 保存结果：最大值的索引 (int64)
    // 题目要求 max_idx 是 tensor，所以要写入它的 data()
    int64_t* idx_ptr = reinterpret_cast<int64_t*>(max_idx->data());
    *idx_ptr = max_i;

    // 6. 保存结果：最大值本身 (类型要还原回去)
    T* val_ptr = reinterpret_cast<T*>(max_val->data());
    *val_ptr = utils::cast<T>(max_v);
}

// 主入口函数
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 根据输入 Tensor 的数据类型，调用对应的模板函数
    switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            argmax_cpu<float>(max_idx, max_val, vals);
            break;
        case LLAISYS_DTYPE_F16:
            argmax_cpu<fp16_t>(max_idx, max_val, vals);
            break;
        case LLAISYS_DTYPE_BF16:
            argmax_cpu<bf16_t>(max_idx, max_val, vals);
            break;
        default:
            std::cerr << "Argmax not implemented for this dtype: " << vals->dtype() << std::endl;
            // 实际工程中这里应该抛出异常
            break;
    }
}

} // namespace llaisys::ops