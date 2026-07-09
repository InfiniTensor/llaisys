#pragma once

// 修改这里：引用内部的 C++ Tensor 定义，而不是不存在的 itensor.h
// 相对路径：src/ops/ -> src/tensor/tensor.hpp
#include "../tensor/tensor.hpp" 

namespace llaisys::ops {

// 2.1 Argmax
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);

// 2.2 Embedding
void embedding(tensor_t out, tensor_t index, tensor_t weight);

// 2.3 Linear
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);

// 2.4 RMS Norm
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);

// 2.5 RoPE
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);

// 2.6 Self Attention
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);

// 2.7 SwiGLU
void swiglu(tensor_t out, tensor_t gate, tensor_t up);

// 3.x Random Sample
int64_t sample(tensor_t logits, float temperature, int top_k, float top_p, uint64_t seed);

// 基础算子
// 确保你有 add 算子 (通常在 src/ops/add/op.cpp 中实现，如果作业1做过)
// 如果没有实现 add，你需要在这里声明并去实现它，或者暂时注释掉(但推理会报错)
void add(tensor_t c, tensor_t a, tensor_t b);
}
 // namespace llaisys::ops
