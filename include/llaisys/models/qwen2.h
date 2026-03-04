#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {

    // Qwen2 模型的基本配置信息
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;          // 权重数据类型
        size_t nlayer;                    // Transformer 层数
        size_t hs;                        // 隐藏层维度 (hidden size)
        size_t nh;                        // 注意力头数 (Q)
        size_t nkvh;                      // KV 头数（用于 GQA/MQA）
        size_t dh;                        // 每个注意力头的维度
        size_t di;                        // MLP 中间层维度
        size_t maxseq;                    // 最大支持序列长度
        size_t voc;                       // 词表大小
        float epsilon;                    // LayerNorm 中的 epsilon
        float theta;                      // RoPE 旋转位置编码的 base 值
        int64_t end_token;                // 结束生成的特殊 token ID
    };

    // Qwen2 模型的所有可训练权重
    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;         // 输入嵌入矩阵
        llaisysTensor_t out_embed;        // 输出嵌入矩阵（通常与输入共享）
        llaisysTensor_t out_norm_w;       // 最终 LayerNorm 的缩放参数

        // 每层的注意力模块权重（数组长度 = nlayer）
        llaisysTensor_t* attn_norm_w;     // Attention 前的 LayerNorm
        llaisysTensor_t* attn_q_w;        // Q 投影矩阵
        llaisysTensor_t* attn_q_b;        // Q 偏置（若存在）
        llaisysTensor_t* attn_k_w;        // K 投影矩阵
        llaisysTensor_t* attn_k_b;        // K 偏置
        llaisysTensor_t* attn_v_w;        // V 投影矩阵
        llaisysTensor_t* attn_v_b;        // V 偏置
        llaisysTensor_t* attn_o_w;        // 输出投影矩阵

        // 每层的 MLP 模块权重（数组长度 = nlayer）
        llaisysTensor_t* mlp_norm_w;      // MLP 前的 LayerNorm
        llaisysTensor_t* mlp_gate_w;      // SwiGLU 门控权重
        llaisysTensor_t* mlp_up_w;        // SwiGLU 上投影
        llaisysTensor_t* mlp_down_w;      // SwiGLU 下投影
    };

    // 控制文本生成行为的采样超参数
    struct LlaisysSamplingParams {
        int32_t top_k;        // Top-k 采样：<=1 表示使用贪心解码
        float top_p;          // Top-p (nucleus) 采样阈值，(0,1] 有效，<=0 禁用
        float temperature;    // 温度参数，>0 启用，<=0 视为 1.0（无缩放）
        uint32_t seed;        // 随机种子，0 表示使用系统随机源
    };

    // 不透明指针，表示 Qwen2 模型实例
    struct LlaisysQwen2Model;

    // 创建 Qwen2 模型实例，需提供元数据、设备类型及设备 ID 列表
    __export struct LlaisysQwen2Model* llaisysQwen2ModelCreate(
        const struct LlaisysQwen2Meta* meta,
        llaisysDeviceType_t device,
        int* device_ids,
        int ndevice);

    // 释放 Qwen2 模型实例及其占用的所有资源
    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model* model);

    // 获取模型内部权重结构的可写指针，用于加载权重
    __export struct LlaisysQwen2Weights* llaisysQwen2ModelWeights(struct LlaisysQwen2Model* model);

    // 执行完整推理（兼容旧接口，推荐使用 Prefill + Step）
    __export int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model* model,
        int64_t* token_ids,
        size_t ntoken);

    // 执行预填充阶段（处理 prompt，初始化 KV Cache）
    __export int64_t llaisysQwen2ModelPrefill(
        struct LlaisysQwen2Model* model,
        int64_t* token_ids,
        size_t ntoken);

    // 执行单步自回归解码（基于已有 KV Cache 生成下一个 token）
    __export int64_t llaisysQwen2ModelStep(
        struct LlaisysQwen2Model* model,
        int64_t* token_ids,
        size_t ntoken);

    // 带采样参数的推理接口（通过结构体传参）
    __export int64_t llaisysQwen2ModelInferSampling(
        struct LlaisysQwen2Model* model,
        int64_t* token_ids,
        size_t ntoken,
        const struct LlaisysSamplingParams* params);

    // 带采样参数的推理接口（通过独立参数传参）
    __export int64_t llaisysQwen2ModelInferSamplingEx(
        struct LlaisysQwen2Model* model,
        int64_t* token_ids,
        size_t ntoken,
        int32_t top_k,
        float top_p,
        float temperature,
        uint32_t seed);

    // 清空模型内部的 KV 缓存，用于开始新一轮生成
    __export void llaisysQwen2ModelResetKVCache(struct LlaisysQwen2Model* model);

    // 动态启用或禁用 KV 缓存功能（调试或特殊用途）
    __export void llaisysQwen2ModelSetKVCacheEnabled(struct LlaisysQwen2Model* model, uint8_t enabled);

} // __C

#endif // LLAISYS_MODELS_QWEN2_H