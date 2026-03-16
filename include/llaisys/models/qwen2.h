#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

LLAISYS_EXTERN_C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        size_t max_batch_size;  /* 连续批处理：KV-Cache 槽位数，1=单序列（默认） */
        int tp_rank;            /* 张量并行 rank，0..tp_world_size-1；默认 0 */
        int tp_world_size;      /* 张量并行 world size，1=非分布式；默认 1 */
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    /** 将输出层权重（out_norm_w、out_embed）拷到 CPU 并缓存；GPU 推理时最后一层在 CPU 上算以规避 GPU 输出层异常。应在权重加载完成后调用一次。 */
    __export void llaisysQwen2ModelCacheOutputLayerOnCPU(struct LlaisysQwen2Model * model);

    /** 将所有权重与 KV cache 拷到 CPU 并缓存；GPU 推理时整次前向在 CPU 上执行以规避 GPU 算子异常。应在权重加载完成后调用一次。 */
    __export void llaisysQwen2ModelCacheAllWeightsOnCPU(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    /** 返回当前已写入 cache 的长度（prefill/suffix prefill/decode 后更新） */
    __export size_t llaisysQwen2ModelGetCacheLen(struct LlaisysQwen2Model * model);

    /** 返回存储前缀长度为 prefix_len 的 KV cache 所需字节数（供 Export/Import 分配缓冲区） */
    __export size_t llaisysQwen2ModelGetKVCacheBytes(struct LlaisysQwen2Model * model, size_t prefix_len);

    /** 将当前 cache 内容导出到 ptr_out（调用方需预先分配 GetKVCacheBytes(model, cache_len) 字节） */
    __export void llaisysQwen2ModelExportKVCache(struct LlaisysQwen2Model * model, void * ptr_out);

    /** 从 ptr_in 导入前缀长度为 prefix_len 的 KV，并设 cache_len = prefix_len；之后可做 suffix prefill */
    __export void llaisysQwen2ModelImportKVCache(struct LlaisysQwen2Model * model, const void * ptr_in, size_t prefix_len);

    /** 将 cache_len 置 0，用于新请求全量 prefill 前清掉上一轮状态（单 slot 或 slot_id=0） */
    __export void llaisysQwen2ModelResetKVCache(struct LlaisysQwen2Model * model);

    /** 将指定 slot 的 cache_len 置 0；仅当 meta.max_batch_size > 1 时有效 */
    __export void llaisysQwen2ModelResetKVCacheSlot(struct LlaisysQwen2Model * model, size_t slot_id);

    /** 将指定 slot 的 KV cache 导出到 ptr_out（调用方需分配 GetKVCacheBytes(model, GetCacheLenSlot(model, slot_id)) 字节） */
    __export void llaisysQwen2ModelExportKVCacheSlot(struct LlaisysQwen2Model * model, size_t slot_id, void * ptr_out);

    /** 从 ptr_in 导入前缀长度为 prefix_len 的 KV 到指定 slot，并设该 slot 的 cache_len = prefix_len；之后可做 suffix prefill */
    __export void llaisysQwen2ModelImportKVCacheSlot(struct LlaisysQwen2Model * model, size_t slot_id, const void * ptr_in, size_t prefix_len);

    /** 返回指定 slot 的 cache_len；当 max_batch_size==1 时 slot_id 忽略，返回当前唯一 cache_len */
    __export size_t llaisysQwen2ModelGetCacheLenSlot(struct LlaisysQwen2Model * model, size_t slot_id);

    /**
     * 单步推理（支持多 slot）。
     * 当 meta.max_batch_size==1 时 slot_id 被忽略，行为与 llaisysQwen2ModelInfer 一致。
     * 当 max_batch_size>1 时，使用指定 slot 的 KV-Cache 进行 prefill/decode，并更新该 slot 的 cache_len。
     */
    __export int64_t llaisysQwen2ModelInferWithSlot(struct LlaisysQwen2Model * model, size_t slot_id, int64_t * token_ids, size_t ntoken, float temperature, int top_k, float top_p, unsigned long long seed);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, float temperature, int top_k, float top_p, unsigned long long seed);

    /**
     * 诊断用：前 (gpu_up_to_layer+1) 层在 GPU 上跑，其余层与输出在 CPU 上跑；需已调用 CacheAllWeightsOnCPU。
     * gpu_up_to_layer < 0：整次前向在 CPU；=0：仅 embedding 在 GPU；=1：embedding+layer0 在 GPU；依此类推。
     * 返回 next_token。用于逐层对比找出首个产生错误结果的 GPU 层。
     */
    __export int64_t llaisysQwen2ModelInferHybrid(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, float temperature, int top_k, float top_p, unsigned long long seed, int gpu_up_to_layer);

    /**
     * 批量 Decode：一次传入多 slot 的当前 token，返回多个 next token。
     * 用于连续批处理调度器；过渡期内部以 for 循环调用单 slot 单 token 推理，后续可替换为真正的 Batched 算子。
     * 要求：n_batch <= meta.max_batch_size，且每个 slot_id 有效；每个 slot 此时应处于 decode 阶段（cache_len > 0）。
     */
    __export void llaisysQwen2ModelBatchedDecode(
        struct LlaisysQwen2Model * model,
        const size_t * slot_ids,      /* 长度为 n_batch，例如 [0, 2, 5] */
        const int64_t * token_ids,    /* 长度为 n_batch，每个 slot 的当前 token */
        size_t n_batch,
        int64_t * out_next_tokens,    /* 长度为 n_batch 的输出 */
        float temperature,
        int top_k,
        float top_p,
        unsigned long long seed);
}
#endif // LLAISYS_MODELS_QWEN2_H
