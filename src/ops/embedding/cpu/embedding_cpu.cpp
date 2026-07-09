#include "embedding_cpu.hpp"
#include "../../../utils/types.hpp"
#include <cstring> // 用于 std::memcpy (这是内存拷贝神器)

namespace llaisys::ops::cpu {

// 模板函数
template<typename T>
void embedding_kernel(T *out, const int64_t *index, const T *weight,
                      size_t num_tokens, size_t vocab_size, size_t hidden_size) {
    // 遍历每一个 token
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(num_tokens); ++i) {
        // 获取当前要查的词 ID
        int64_t idx = index[i];
        
        // 简单检查一下越界 (虽然题目没强制要求，但为了安全)
        if (idx < 0 || (size_t)idx >= vocab_size) {
            // 在实际工程中这里应该报错，作业里我们暂时跳过或默认全0
            continue; 
        }

        // 计算源地址：weight 的第 idx 行
        const T *src_row = weight + idx * hidden_size;
        
        // 计算目标地址：out 的第 i 行
        T *dst_row = out + i * hidden_size;

        // 核心操作：直接内存拷贝一行 (速度最快)
        // copy 的字节数 = hidden_size * sizeof(T)
        std::memcpy(dst_row, src_row, hidden_size * sizeof(T));
    }
}

// 入口函数
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t num_tokens, size_t vocab_size, size_t hidden_size,
               llaisysDataType_t dtype) {
    
    // index 已经被强制要求为 Int64，所以强转为 int64_t*
    const int64_t *idx_ptr = (const int64_t*)index;

    // 根据数据类型分发
    if (dtype == LLAISYS_DTYPE_F32) {
        embedding_kernel<float>(
            (float*)out, idx_ptr, (const float*)weight, 
            num_tokens, vocab_size, hidden_size
        );
    } 
    else if (dtype == LLAISYS_DTYPE_F16) {
        // fp16 本质上只是搬运内存，不需要数值计算，所以只要结构体大小对就行
        // 使用 llaisys::fp16_t
        embedding_kernel<llaisys::fp16_t>(
            (llaisys::fp16_t*)out, idx_ptr, (const llaisys::fp16_t*)weight, 
            num_tokens, vocab_size, hidden_size
        );
    }
    else if (dtype == LLAISYS_DTYPE_BF16) {
        // bf16 同理
        embedding_kernel<llaisys::bf16_t>(
            (llaisys::bf16_t*)out, idx_ptr, (const llaisys::bf16_t*)weight, 
            num_tokens, vocab_size, hidden_size
        );
    }
}

} // namespace llaisys::ops::cpu
