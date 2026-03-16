#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
/**
 * 从 logits 按概率随机采样一个 token 索引。
 * 支持 Temperature、Top-K、Top-P（nucleus）采样。
 *
 * @param out_idx 输出张量，shape=[1], dtype=int64，写入采样得到的 token id
 * @param logits  一维 logits，dtype 支持 f32/f16/bf16
 * @param temperature 温度，<=0 或极小值时退化为 argmax
 * @param top_k 保留概率最高的 k 个 token，<=0 表示不限制
 * @param top_p nucleus 采样累积概率阈值，<=0 或 >=1 表示不限制
 * @param seed 随机种子，0 表示使用随机设备
 */
void sample(tensor_t out_idx, tensor_t logits, float temperature, int top_k,
            float top_p, uint64_t seed);
}
