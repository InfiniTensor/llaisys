/**
 * Qwen2 模型推理的 C++ 实现：模型创建、权重分配、单步前向与 KV Cache 管理。
 *
 * 对外暴露 C 接口（见 include/llaisys/models/qwen2.h）：
 * - llaisysQwen2ModelCreate / Destroy / Weights：创建、销毁、获取权重句柄；
 * - llaisysQwen2ModelInfer：给定当前 token 序列，执行一次前向，返回下一个 token id。
 *
 * 前向流程：embedding -> 逐层 Transformer Block（attention + MLP）-> 最后一层 norm -> 输出层 linear -> argmax。
 */
#include "llaisys/llaisys_tensor.hpp"
#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"
#ifdef ENABLE_NCCL
#include "llaisys/nccl_comm.h"
#endif

#include "../core/llaisys_core.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/sample/op.hpp"
#include "../tensor/tensor.hpp"
#include "../utils.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace {

using namespace llaisys;
using tensor_t = llaisys::tensor_t;

/// 从 C 接口的 LlaisysTensor 包装中取出内部 tensor_t 指针
inline tensor_t get_t(llaisysTensor_t t) { return t->tensor; }

/// 按当前设备做同步内存拷贝（H2H 或 D2D），供写 KV cache 和写回 hidden 用
void copy_sync(void *dst, const void *src, size_t bytes, llaisysDeviceType_t dev) {
    llaisys::core::context().setDevice(dev, 0);
    llaisysMemcpyKind_t kind = (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;
    llaisys::core::context().runtime().api()->memcpy_sync(dst, src, bytes, kind);
}

/// 设备 -> 主机拷贝（Export 用）；device_id 默认 0，多卡时由调用方传入 model->device_id
void copy_device_to_host(void *host_dst, const void *dev_src, size_t bytes, llaisysDeviceType_t dev, int device_id = 0) {
    if (dev == LLAISYS_DEVICE_CPU) {
        std::memcpy(host_dst, dev_src, bytes);
        return;
    }
    llaisys::core::context().setDevice(dev, device_id);
    llaisys::core::context().runtime().api()->memcpy_sync(host_dst, dev_src, bytes, LLAISYS_MEMCPY_D2H);
}

/// 主机 -> 设备拷贝（Import 用）；device_id 默认 0，多卡时由调用方传入 model->device_id
void copy_host_to_device(void *dev_dst, const void *host_src, size_t bytes, llaisysDeviceType_t dev, int device_id = 0) {
    if (dev == LLAISYS_DEVICE_CPU) {
        std::memcpy(dev_dst, host_src, bytes);
        return;
    }
    llaisys::core::context().setDevice(dev, device_id);
    llaisys::core::context().runtime().api()->memcpy_sync(dev_dst, host_src, bytes, LLAISYS_MEMCPY_H2D);
}

} // namespace

/// Qwen2 模型内部表示：元信息、权重张量、每层 K/V cache 及当前已填充长度
/// 当 meta.max_batch_size > 1 时：k/v_caches 每层形状 [max_batch_size, maxseq, nkvh, dh]，cache_lens 长度为 max_batch_size
/// 当 max_batch_size == 1：k/v_caches 每层形状 [maxseq, nkvh, dh]，cache_lens.size()==1
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    std::vector<tensor_t> k_caches;   /// 每层 key cache；单 slot [maxseq,nkvh,dh]，多 slot [max_batch,maxseq,nkvh,dh]
    std::vector<tensor_t> v_caches;   /// 每层 value cache，同上
    std::vector<size_t> cache_lens;   /// 每 slot 已写入长度；单 slot 时 size()==1
    llaisysDeviceType_t device_type;
    int device_id;
    tensor_t out_norm_w_cpu;   /// 可选：输出层 norm 权重的 CPU 副本，GPU 时最后一层在 CPU 算用
    tensor_t out_embed_cpu;    /// 可选：输出层 linear 权重的 CPU 副本
    /// 全量 CPU 缓存：若 in_embed_cpu 非空则推理时整次前向在 CPU 上执行
    tensor_t in_embed_cpu;
    std::vector<tensor_t> attn_norm_w_cpu, attn_q_w_cpu, attn_q_b_cpu, attn_k_w_cpu, attn_k_b_cpu;
    std::vector<tensor_t> attn_v_w_cpu, attn_v_b_cpu, attn_o_w_cpu;
    std::vector<tensor_t> mlp_norm_w_cpu, mlp_gate_w_cpu, mlp_up_w_cpu, mlp_down_w_cpu;
    std::vector<tensor_t> k_caches_cpu, v_caches_cpu;
    /// 张量并行：AllGather 结果缓冲区，仅 tp_world_size>1 且 GPU 时非空
    tensor_t tp_gather_q;   /// [maxseq, nh*dh]
    tensor_t tp_gather_k;   /// [maxseq, nkvh*dh]
    tensor_t tp_gather_v;   /// [maxseq, nkvh*dh]
    tensor_t tp_gather_gate; /// [maxseq, di]
    tensor_t tp_gather_up;   /// [maxseq, di]
};

LLAISYS_EXTERN_C {

/// 根据 meta 为模型分配所有权重张量；tp_world_size>1 时按张量并行分片（列并行：输出维切分；行并行：输入维切分）
static void create_weight_tensors(LlaisysQwen2Model *m) {
    const LlaisysQwen2Meta *meta = &m->meta;
    const size_t nlayer = meta->nlayer;
    const size_t hs = meta->hs;
    const size_t nh = meta->nh;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t di = meta->di;
    const size_t voc = meta->voc;
    const int tp_world = meta->tp_world_size > 0 ? meta->tp_world_size : 1;
    const llaisysDataType_t dtype = meta->dtype;
    const llaisysDeviceType_t dev = m->device_type;
    const int dev_id = m->device_id;

    auto mk = [&](const std::vector<size_t> &shape) {
        return LlaisysTensor{llaisys::Tensor::create(shape, dtype, dev, dev_id)};
    };

    const bool use_tp = (tp_world > 1);
    const size_t W = use_tp ? static_cast<size_t>(tp_world) : 1u;
    const size_t nhdh = nh * dh;
    const size_t nkvhdh = nkvh * dh;

    // 词嵌入与输出层：非 TP 时全量，TP 时也全量复制（每 rank 一份，避免 token 路由）
    m->weights.in_embed = new LlaisysTensor(mk({voc, hs}));
    m->weights.out_embed = new LlaisysTensor(mk({voc, hs}));
    m->weights.out_norm_w = new LlaisysTensor(mk({hs}));

    m->weights.attn_norm_w = new llaisysTensor_t[nlayer];
    m->weights.attn_q_w = new llaisysTensor_t[nlayer];
    m->weights.attn_q_b = new llaisysTensor_t[nlayer];
    m->weights.attn_k_w = new llaisysTensor_t[nlayer];
    m->weights.attn_k_b = new llaisysTensor_t[nlayer];
    m->weights.attn_v_w = new llaisysTensor_t[nlayer];
    m->weights.attn_v_b = new llaisysTensor_t[nlayer];
    m->weights.attn_o_w = new llaisysTensor_t[nlayer];
    m->weights.mlp_norm_w = new llaisysTensor_t[nlayer];
    m->weights.mlp_gate_w = new llaisysTensor_t[nlayer];
    m->weights.mlp_up_w = new llaisysTensor_t[nlayer];
    m->weights.mlp_down_w = new llaisysTensor_t[nlayer];

    for (size_t i = 0; i < nlayer; i++) {
        m->weights.attn_norm_w[i] = new LlaisysTensor(mk({hs}));
        if (use_tp) {
            const size_t nhdh_l = nhdh / W;
            const size_t nkvhdh_l = nkvhdh / W;
            const size_t di_l = di / W;
            m->weights.attn_q_w[i] = new LlaisysTensor(mk({nhdh_l, hs}));
            m->weights.attn_q_b[i] = new LlaisysTensor(mk({nhdh_l}));
            m->weights.attn_k_w[i] = new LlaisysTensor(mk({nkvhdh_l, hs}));
            m->weights.attn_k_b[i] = new LlaisysTensor(mk({nkvhdh_l}));
            m->weights.attn_v_w[i] = new LlaisysTensor(mk({nkvhdh_l, hs}));
            m->weights.attn_v_b[i] = new LlaisysTensor(mk({nkvhdh_l}));
            m->weights.attn_o_w[i] = new LlaisysTensor(mk({hs, nhdh_l}));
            m->weights.mlp_norm_w[i] = new LlaisysTensor(mk({hs}));
            m->weights.mlp_gate_w[i] = new LlaisysTensor(mk({di_l, hs}));
            m->weights.mlp_up_w[i] = new LlaisysTensor(mk({di_l, hs}));
            m->weights.mlp_down_w[i] = new LlaisysTensor(mk({hs, di_l}));
        } else {
            m->weights.attn_q_w[i] = new LlaisysTensor(mk({nhdh, hs}));
            m->weights.attn_q_b[i] = new LlaisysTensor(mk({nhdh}));
            m->weights.attn_k_w[i] = new LlaisysTensor(mk({nkvhdh, hs}));
            m->weights.attn_k_b[i] = new LlaisysTensor(mk({nkvhdh}));
            m->weights.attn_v_w[i] = new LlaisysTensor(mk({nkvhdh, hs}));
            m->weights.attn_v_b[i] = new LlaisysTensor(mk({nkvhdh}));
            m->weights.attn_o_w[i] = new LlaisysTensor(mk({hs, nhdh}));
            m->weights.mlp_norm_w[i] = new LlaisysTensor(mk({hs}));
            m->weights.mlp_gate_w[i] = new LlaisysTensor(mk({di, hs}));
            m->weights.mlp_up_w[i] = new LlaisysTensor(mk({di, hs}));
            m->weights.mlp_down_w[i] = new LlaisysTensor(mk({hs, di}));
        }
    }
}

/// 创建模型：分配 meta、权重张量、每层 KV cache，返回模型指针；权重数据由调用方通过 llaisysQwen2ModelWeights + tensorLoad 写入
struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                  llaisysDeviceType_t device,
                                                  int *device_ids,
                                                  int ndevice) {
    (void)device_ids;
    (void)ndevice;
    LlaisysQwen2Model *m = new LlaisysQwen2Model();
    m->meta = *meta;
    if (m->meta.max_batch_size == 0)
        m->meta.max_batch_size = 1;
    const size_t max_batch = m->meta.max_batch_size;
    m->cache_lens.assign(max_batch, 0);
    m->device_type = device;
    m->device_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;

    llaisys::core::context().setDevice(device, m->device_id);

    create_weight_tensors(m);

    const size_t nlayer = meta->nlayer;
    const size_t maxseq = meta->maxseq;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;

    m->out_norm_w_cpu = nullptr;
    m->out_embed_cpu = nullptr;
    m->in_embed_cpu = nullptr;

    m->k_caches.resize(nlayer);
    m->v_caches.resize(nlayer);
    if (max_batch > 1) {
        for (size_t i = 0; i < nlayer; i++) {
            m->k_caches[i] = llaisys::Tensor::create(
                {max_batch, maxseq, nkvh, dh}, meta->dtype, device, m->device_id);
            m->v_caches[i] = llaisys::Tensor::create(
                {max_batch, maxseq, nkvh, dh}, meta->dtype, device, m->device_id);
        }
    } else {
        for (size_t i = 0; i < nlayer; i++) {
            m->k_caches[i] = llaisys::Tensor::create(
                {maxseq, nkvh, dh}, meta->dtype, device, m->device_id);
            m->v_caches[i] = llaisys::Tensor::create(
                {maxseq, nkvh, dh}, meta->dtype, device, m->device_id);
        }
    }

    if (meta->tp_world_size > 1 && device == LLAISYS_DEVICE_NVIDIA) {
        const size_t nhdh = meta->nh * meta->dh;
        const size_t nkvhdh = meta->nkvh * meta->dh;
        m->tp_gather_q = llaisys::Tensor::create(
            {maxseq, nhdh}, meta->dtype, device, m->device_id);
        m->tp_gather_k = llaisys::Tensor::create(
            {maxseq, nkvhdh}, meta->dtype, device, m->device_id);
        m->tp_gather_v = llaisys::Tensor::create(
            {maxseq, nkvhdh}, meta->dtype, device, m->device_id);
        m->tp_gather_gate = llaisys::Tensor::create(
            {maxseq, meta->di}, meta->dtype, device, m->device_id);
        m->tp_gather_up = llaisys::Tensor::create(
            {maxseq, meta->di}, meta->dtype, device, m->device_id);
    } else {
        m->tp_gather_q = nullptr;
        m->tp_gather_k = nullptr;
        m->tp_gather_v = nullptr;
        m->tp_gather_gate = nullptr;
        m->tp_gather_up = nullptr;
    }
    return m;
}

/// 释放模型：先销毁所有权重张量和指针数组，再 delete 模型本体
void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) return;
    const size_t nlayer = model->meta.nlayer;

    tensorDestroy(model->weights.in_embed);
    tensorDestroy(model->weights.out_embed);
    tensorDestroy(model->weights.out_norm_w);

    for (size_t i = 0; i < nlayer; i++) {
        tensorDestroy(model->weights.attn_norm_w[i]);
        tensorDestroy(model->weights.attn_q_w[i]);
        tensorDestroy(model->weights.attn_q_b[i]);
        tensorDestroy(model->weights.attn_k_w[i]);
        tensorDestroy(model->weights.attn_k_b[i]);
        tensorDestroy(model->weights.attn_v_w[i]);
        tensorDestroy(model->weights.attn_v_b[i]);
        tensorDestroy(model->weights.attn_o_w[i]);
        tensorDestroy(model->weights.mlp_norm_w[i]);
        tensorDestroy(model->weights.mlp_gate_w[i]);
        tensorDestroy(model->weights.mlp_up_w[i]);
        tensorDestroy(model->weights.mlp_down_w[i]);
    }
    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;

    if (model->out_norm_w_cpu) model->out_norm_w_cpu = nullptr;
    if (model->out_embed_cpu) model->out_embed_cpu = nullptr;
    if (model->in_embed_cpu) model->in_embed_cpu = nullptr;
    model->attn_norm_w_cpu.clear();
    model->attn_q_w_cpu.clear();
    model->attn_q_b_cpu.clear();
    model->attn_k_w_cpu.clear();
    model->attn_k_b_cpu.clear();
    model->attn_v_w_cpu.clear();
    model->attn_v_b_cpu.clear();
    model->attn_o_w_cpu.clear();
    model->mlp_norm_w_cpu.clear();
    model->mlp_gate_w_cpu.clear();
    model->mlp_up_w_cpu.clear();
    model->mlp_down_w_cpu.clear();
    model->k_caches_cpu.clear();
    model->v_caches_cpu.clear();
    if (model->tp_gather_q) model->tp_gather_q = nullptr;
    if (model->tp_gather_k) model->tp_gather_k = nullptr;
    if (model->tp_gather_v) model->tp_gather_v = nullptr;
    if (model->tp_gather_gate) model->tp_gather_gate = nullptr;
    if (model->tp_gather_up) model->tp_gather_up = nullptr;

    delete model;
}

void llaisysQwen2ModelCacheOutputLayerOnCPU(struct LlaisysQwen2Model *model) {
    if (!model || model->device_type == LLAISYS_DEVICE_CPU) return;
    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t hs = meta->hs, voc = meta->voc;
    const llaisysDataType_t dtype = meta->dtype;
    const size_t elem_size = llaisys::utils::dsize(dtype);

    llaisys::core::context().setDevice(model->device_type, model->device_id);
    llaisys::core::context().runtime().api()->device_synchronize();

    tensor_t in_embed_cpu = llaisys::Tensor::create({voc, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
    copy_device_to_host(in_embed_cpu->data(), get_t(model->weights.in_embed)->data(), voc * hs * elem_size, model->device_type, model->device_id);
    model->in_embed_cpu = in_embed_cpu;

    tensor_t norm_cpu = llaisys::Tensor::create({hs}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t embed_cpu = llaisys::Tensor::create({voc, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
    copy_device_to_host(norm_cpu->data(), get_t(model->weights.out_norm_w)->data(), hs * elem_size, model->device_type, model->device_id);
    copy_device_to_host(embed_cpu->data(), get_t(model->weights.out_embed)->data(), voc * hs * elem_size, model->device_type, model->device_id);

    model->out_norm_w_cpu = norm_cpu;
    model->out_embed_cpu = embed_cpu;
}

void llaisysQwen2ModelCacheAllWeightsOnCPU(struct LlaisysQwen2Model *model) {
    if (!model || model->device_type == LLAISYS_DEVICE_CPU) return;
    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh, di = meta->di, voc = meta->voc;
    const llaisysDataType_t dtype = meta->dtype;
    const size_t elem_size = llaisys::utils::dsize(dtype);
    llaisys::core::context().setDevice(model->device_type, model->device_id);
    llaisys::core::context().runtime().api()->device_synchronize();

    tensor_t in_embed_cpu = llaisys::Tensor::create({voc, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
    copy_device_to_host(in_embed_cpu->data(), get_t(model->weights.in_embed)->data(), voc * hs * elem_size, model->device_type, model->device_id);
    if (!model->out_norm_w_cpu) {
        model->out_norm_w_cpu = llaisys::Tensor::create({hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        model->out_embed_cpu = llaisys::Tensor::create({voc, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
    }
    copy_device_to_host(model->out_norm_w_cpu->data(), get_t(model->weights.out_norm_w)->data(), hs * elem_size, model->device_type, model->device_id);
    copy_device_to_host(model->out_embed_cpu->data(), get_t(model->weights.out_embed)->data(), voc * hs * elem_size, model->device_type, model->device_id);

    model->attn_norm_w_cpu.resize(nlayer);
    model->attn_q_w_cpu.resize(nlayer);
    model->attn_q_b_cpu.resize(nlayer);
    model->attn_k_w_cpu.resize(nlayer);
    model->attn_k_b_cpu.resize(nlayer);
    model->attn_v_w_cpu.resize(nlayer);
    model->attn_v_b_cpu.resize(nlayer);
    model->attn_o_w_cpu.resize(nlayer);
    model->mlp_norm_w_cpu.resize(nlayer);
    model->mlp_gate_w_cpu.resize(nlayer);
    model->mlp_up_w_cpu.resize(nlayer);
    model->mlp_down_w_cpu.resize(nlayer);
    for (size_t i = 0; i < nlayer; i++) {
        model->attn_norm_w_cpu[i] = llaisys::Tensor::create({hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->attn_norm_w_cpu[i]->data(), get_t(model->weights.attn_norm_w[i])->data(), hs * elem_size, model->device_type, model->device_id);
        model->attn_q_w_cpu[i] = llaisys::Tensor::create({nh * dh, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->attn_q_w_cpu[i]->data(), get_t(model->weights.attn_q_w[i])->data(), nh * dh * hs * elem_size, model->device_type, model->device_id);
        model->attn_q_b_cpu[i] = llaisys::Tensor::create({nh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->attn_q_b_cpu[i]->data(), get_t(model->weights.attn_q_b[i])->data(), nh * dh * elem_size, model->device_type, model->device_id);
        model->attn_k_w_cpu[i] = llaisys::Tensor::create({nkvh * dh, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->attn_k_w_cpu[i]->data(), get_t(model->weights.attn_k_w[i])->data(), nkvh * dh * hs * elem_size, model->device_type, model->device_id);
        model->attn_k_b_cpu[i] = llaisys::Tensor::create({nkvh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->attn_k_b_cpu[i]->data(), get_t(model->weights.attn_k_b[i])->data(), nkvh * dh * elem_size, model->device_type, model->device_id);
        model->attn_v_w_cpu[i] = llaisys::Tensor::create({nkvh * dh, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->attn_v_w_cpu[i]->data(), get_t(model->weights.attn_v_w[i])->data(), nkvh * dh * hs * elem_size, model->device_type, model->device_id);
        model->attn_v_b_cpu[i] = llaisys::Tensor::create({nkvh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->attn_v_b_cpu[i]->data(), get_t(model->weights.attn_v_b[i])->data(), nkvh * dh * elem_size, model->device_type, model->device_id);
        model->attn_o_w_cpu[i] = llaisys::Tensor::create({hs, nh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->attn_o_w_cpu[i]->data(), get_t(model->weights.attn_o_w[i])->data(), hs * nh * dh * elem_size, model->device_type, model->device_id);
        model->mlp_norm_w_cpu[i] = llaisys::Tensor::create({hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->mlp_norm_w_cpu[i]->data(), get_t(model->weights.mlp_norm_w[i])->data(), hs * elem_size, model->device_type, model->device_id);
        model->mlp_gate_w_cpu[i] = llaisys::Tensor::create({di, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->mlp_gate_w_cpu[i]->data(), get_t(model->weights.mlp_gate_w[i])->data(), di * hs * elem_size, model->device_type, model->device_id);
        model->mlp_up_w_cpu[i] = llaisys::Tensor::create({di, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->mlp_up_w_cpu[i]->data(), get_t(model->weights.mlp_up_w[i])->data(), di * hs * elem_size, model->device_type, model->device_id);
        model->mlp_down_w_cpu[i] = llaisys::Tensor::create({hs, di}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(model->mlp_down_w_cpu[i]->data(), get_t(model->weights.mlp_down_w[i])->data(), hs * di * elem_size, model->device_type, model->device_id);
    }

    model->k_caches_cpu.resize(nlayer);
    model->v_caches_cpu.resize(nlayer);
    const std::vector<size_t> &k0_shape = model->k_caches[0]->shape();
    for (size_t i = 0; i < nlayer; i++) {
        model->k_caches_cpu[i] = llaisys::Tensor::create(k0_shape, dtype, LLAISYS_DEVICE_CPU, 0);
        model->v_caches_cpu[i] = llaisys::Tensor::create(k0_shape, dtype, LLAISYS_DEVICE_CPU, 0);
    }
    model->in_embed_cpu = in_embed_cpu;
}

/// 返回模型权重结构体指针，供 Python 侧根据 safetensors key 找到对应句柄并 tensorLoad
struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return model ? &model->weights : nullptr;
}

} // extern "C"

namespace {

/**
 * 单层 Transformer Block 前向：Attention（norm -> q/k/v -> RoPE -> 写 KV cache -> attention -> o_proj -> 残差）+ MLP（norm -> gate/up -> SwiGLU -> down -> 残差）。
 * 输入输出通过 hidden 传入并在本函数内原地更新；slot_id 指定使用哪一槽的 KV cache（多 slot 时）。
 */
void forward_layer(LlaisysQwen2Model *m, size_t layer_idx, size_t slot_id,
                   tensor_t hidden,      // [seq, hs]
                   tensor_t normed,      // [seq, hs]
                   tensor_t q_buf,       // [seq, nh*dh]
                   tensor_t k_buf,       // [seq, nkvh*dh]
                   tensor_t v_buf,       // [seq, nkvh*dh]
                   tensor_t q_rope,      // [seq, nh, dh]
                   tensor_t k_rope,      // [seq, nkvh, dh]
                   tensor_t attn_val,    // [seq, nh, dh]
                   tensor_t o_proj_out,  // [seq, hs]
                   tensor_t res_buf,     // [seq, hs]
                   tensor_t gate_buf,    // [seq, di]
                   tensor_t up_buf,      // [seq, di]
                   tensor_t mlp_buf,     // [seq, di]
                   tensor_t down_buf,    // [seq, hs]
                   size_t seq_len, size_t cache_start,
                   tensor_t pos_ids_t) {
    const LlaisysQwen2Meta *meta = &m->meta;
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh;
    const size_t maxseq = meta->maxseq;
    const float eps = meta->epsilon, theta = meta->theta;
    LlaisysQwen2Weights *w = &m->weights;
    tensor_t wt = get_t(w->attn_norm_w[layer_idx]);
    const float scale = 1.f / std::sqrt(static_cast<float>(dh));

    tensor_t k_cache_raw = m->k_caches[layer_idx];
    tensor_t v_cache_raw = m->v_caches[layer_idx];
    tensor_t k_cache, v_cache;
    if (meta->max_batch_size > 1) {
        k_cache = k_cache_raw->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
        v_cache = v_cache_raw->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
    } else {
        k_cache = k_cache_raw;
        v_cache = v_cache_raw;
    }

    // ---------- Attention 分支：norm -> Q/K/V 投影 -> RoPE -> 写 KV cache -> causal attention -> o_proj -> 残差 ----------
    llaisys::ops::rms_norm(normed, hidden, wt, eps);
    llaisys::ops::linear(q_buf, normed, get_t(w->attn_q_w[layer_idx]), get_t(w->attn_q_b[layer_idx]));
    llaisys::ops::linear(k_buf, normed, get_t(w->attn_k_w[layer_idx]), get_t(w->attn_k_b[layer_idx]));
    llaisys::ops::linear(v_buf, normed, get_t(w->attn_v_w[layer_idx]), get_t(w->attn_v_b[layer_idx]));

    // 展平为 [seq, nh, dh] / [seq, nkvh, dh] 以做 RoPE
    std::vector<size_t> shape_q = {seq_len, nh, dh};
    std::vector<size_t> shape_kv = {seq_len, nkvh, dh};
    tensor_t q_view = q_buf->view(shape_q);
    tensor_t k_view = k_buf->view(shape_kv);
    tensor_t v_view = v_buf->view(shape_kv);

    llaisys::ops::rope(q_rope, q_view, pos_ids_t, theta);
    llaisys::ops::rope(k_rope, k_view, pos_ids_t, theta);

    // 将本步的 K/V 写入 cache 的 [cache_start, cache_start+seq_len) 位置，decode 时复用历史
    const size_t elem_size = llaisys::utils::dsize(meta->dtype);
    const size_t kv_row_bytes = nkvh * dh * elem_size;
    for (size_t s = 0; s < seq_len; s++) {
        size_t cache_pos = cache_start + s;
        copy_sync(
            reinterpret_cast<std::byte *>(k_cache->data()) + cache_pos * kv_row_bytes,
            reinterpret_cast<const std::byte *>(k_rope->data()) + s * kv_row_bytes,
            kv_row_bytes, m->device_type);
        copy_sync(
            reinterpret_cast<std::byte *>(v_cache->data()) + cache_pos * kv_row_bytes,
            reinterpret_cast<const std::byte *>(v_buf->data()) + s * kv_row_bytes,
            kv_row_bytes, m->device_type);
    }

    // 当前有效长度为 cache_start + seq_len；取 cache 的该前缀做 attention
    size_t kv_len = cache_start + seq_len;
    tensor_t k_slice = k_cache->slice(0, 0, kv_len);
    tensor_t v_slice = v_cache->slice(0, 0, kv_len);

    llaisys::ops::self_attention(attn_val, q_rope, k_slice, v_slice, scale);

    std::vector<size_t> shape_attn_flat = {seq_len, nh * dh};
    tensor_t attn_flat = attn_val->view(shape_attn_flat);
    llaisys::ops::linear(o_proj_out, attn_flat, get_t(w->attn_o_w[layer_idx]), nullptr);

    llaisys::ops::add(res_buf, hidden, o_proj_out);
    copy_sync(hidden->data(), res_buf->data(), seq_len * hs * elem_size, m->device_type);

    // ---------- MLP 分支：norm -> gate/up -> SwiGLU -> down -> 残差 ----------
    llaisys::ops::rms_norm(normed, hidden, get_t(w->mlp_norm_w[layer_idx]), eps);
    llaisys::ops::linear(gate_buf, normed, get_t(w->mlp_gate_w[layer_idx]), nullptr);
    llaisys::ops::linear(up_buf, normed, get_t(w->mlp_up_w[layer_idx]), nullptr);
    llaisys::ops::swiglu(mlp_buf, gate_buf, up_buf);
    llaisys::ops::linear(down_buf, mlp_buf, get_t(w->mlp_down_w[layer_idx]), nullptr);
    llaisys::ops::add(res_buf, hidden, down_buf);
    copy_sync(hidden->data(), res_buf->data(), seq_len * hs * elem_size, m->device_type);
}

/// 单层 Transformer 前向，使用 CPU 权重与 CPU KV cache（全量 CPU 推理用）
void forward_layer_cpu(LlaisysQwen2Model *m, size_t layer_idx, size_t slot_id,
                       tensor_t hidden, tensor_t normed, tensor_t q_buf, tensor_t k_buf, tensor_t v_buf,
                       tensor_t q_rope, tensor_t k_rope, tensor_t attn_val, tensor_t o_proj_out, tensor_t res_buf,
                       tensor_t gate_buf, tensor_t up_buf, tensor_t mlp_buf, tensor_t down_buf,
                       size_t seq_len, size_t cache_start, tensor_t pos_ids_t) {
    const LlaisysQwen2Meta *meta = &m->meta;
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh, maxseq = meta->maxseq;
    const float eps = meta->epsilon, theta = meta->theta;
    const size_t elem_size = llaisys::utils::dsize(meta->dtype);
    const float scale = 1.f / std::sqrt(static_cast<float>(dh));

    tensor_t k_cache = m->k_caches_cpu[layer_idx];
    tensor_t v_cache = m->v_caches_cpu[layer_idx];
    if (meta->max_batch_size > 1) {
        k_cache = k_cache->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
        v_cache = v_cache->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
    }

    llaisys::ops::rms_norm(normed, hidden, m->attn_norm_w_cpu[layer_idx], eps);
    llaisys::ops::linear(q_buf, normed, m->attn_q_w_cpu[layer_idx], m->attn_q_b_cpu[layer_idx]);
    llaisys::ops::linear(k_buf, normed, m->attn_k_w_cpu[layer_idx], m->attn_k_b_cpu[layer_idx]);
    llaisys::ops::linear(v_buf, normed, m->attn_v_w_cpu[layer_idx], m->attn_v_b_cpu[layer_idx]);

    std::vector<size_t> shape_q = {seq_len, nh, dh};
    std::vector<size_t> shape_kv = {seq_len, nkvh, dh};
    tensor_t q_view = q_buf->view(shape_q);
    tensor_t k_view = k_buf->view(shape_kv);
    tensor_t v_view = v_buf->view(shape_kv);
    llaisys::ops::rope(q_rope, q_view, pos_ids_t, theta);
    llaisys::ops::rope(k_rope, k_view, pos_ids_t, theta);

    const size_t kv_row_bytes = nkvh * dh * elem_size;
    for (size_t s = 0; s < seq_len; s++) {
        size_t cache_pos = cache_start + s;
        copy_sync(
            reinterpret_cast<std::byte *>(k_cache->data()) + cache_pos * kv_row_bytes,
            reinterpret_cast<const std::byte *>(k_rope->data()) + s * kv_row_bytes,
            kv_row_bytes, LLAISYS_DEVICE_CPU);
        copy_sync(
            reinterpret_cast<std::byte *>(v_cache->data()) + cache_pos * kv_row_bytes,
            reinterpret_cast<const std::byte *>(v_buf->data()) + s * kv_row_bytes,
            kv_row_bytes, LLAISYS_DEVICE_CPU);
    }

    size_t kv_len = cache_start + seq_len;
    tensor_t k_slice = k_cache->slice(0, 0, kv_len);
    tensor_t v_slice = v_cache->slice(0, 0, kv_len);
    llaisys::ops::self_attention(attn_val, q_rope, k_slice, v_slice, scale);

    std::vector<size_t> shape_attn_flat = {seq_len, nh * dh};
    tensor_t attn_flat = attn_val->view(shape_attn_flat);
    llaisys::ops::linear(o_proj_out, attn_flat, m->attn_o_w_cpu[layer_idx], nullptr);
    llaisys::ops::add(res_buf, hidden, o_proj_out);
    copy_sync(hidden->data(), res_buf->data(), seq_len * hs * elem_size, LLAISYS_DEVICE_CPU);

    llaisys::ops::rms_norm(normed, hidden, m->mlp_norm_w_cpu[layer_idx], eps);
    llaisys::ops::linear(gate_buf, normed, m->mlp_gate_w_cpu[layer_idx], nullptr);
    llaisys::ops::linear(up_buf, normed, m->mlp_up_w_cpu[layer_idx], nullptr);
    llaisys::ops::swiglu(mlp_buf, gate_buf, up_buf);
    llaisys::ops::linear(down_buf, mlp_buf, m->mlp_down_w_cpu[layer_idx], nullptr);
    llaisys::ops::add(res_buf, hidden, down_buf);
    copy_sync(hidden->data(), res_buf->data(), seq_len * hs * elem_size, LLAISYS_DEVICE_CPU);
}

#ifdef ENABLE_NCCL
/// 张量并行单层前向：列并行 Q/K/V/Gate/Up + AllGather，行并行 O/Down + AllReduce(Sum)。stream 为 CUDA 流。
void forward_layer_tp(LlaisysQwen2Model *m, size_t layer_idx, size_t slot_id,
                     tensor_t hidden, tensor_t normed,
                     tensor_t q_buf_local, tensor_t k_buf_local, tensor_t v_buf_local,
                     tensor_t q_rope, tensor_t k_rope, tensor_t attn_val, tensor_t o_proj_out, tensor_t res_buf,
                     tensor_t gate_buf_local, tensor_t up_buf_local, tensor_t mlp_buf, tensor_t down_buf,
                     size_t seq_len, size_t cache_start, tensor_t pos_ids_t, void *stream) {
    const LlaisysQwen2Meta *meta = &m->meta;
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh, di = meta->di;
    const size_t maxseq = meta->maxseq;
    const float eps = meta->epsilon, theta = meta->theta;
    const llaisysDataType_t dtype = meta->dtype;
    const size_t elem_size = llaisys::utils::dsize(dtype);
    const float scale = 1.f / std::sqrt(static_cast<float>(dh));
    const int W = meta->tp_world_size;
    const size_t nhdh = nh * dh;
    const size_t nkvhdh = nkvh * dh;
    const size_t nhdh_l = nhdh / static_cast<size_t>(W);
    const size_t nkvhdh_l = nkvhdh / static_cast<size_t>(W);
    const size_t di_l = di / static_cast<size_t>(W);

    LlaisysQwen2Weights *w = &m->weights;
    tensor_t wt = get_t(w->attn_norm_w[layer_idx]);
    tensor_t k_cache_raw = m->k_caches[layer_idx];
    tensor_t v_cache_raw = m->v_caches[layer_idx];
    tensor_t k_cache, v_cache;
    if (meta->max_batch_size > 1) {
        k_cache = k_cache_raw->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
        v_cache = v_cache_raw->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
    } else {
        k_cache = k_cache_raw;
        v_cache = v_cache_raw;
    }

    llaisys::ops::rms_norm(normed, hidden, wt, eps);
    llaisys::ops::linear(q_buf_local, normed, get_t(w->attn_q_w[layer_idx]), get_t(w->attn_q_b[layer_idx]));
    llaisys::ops::linear(k_buf_local, normed, get_t(w->attn_k_w[layer_idx]), get_t(w->attn_k_b[layer_idx]));
    llaisys::ops::linear(v_buf_local, normed, get_t(w->attn_v_w[layer_idx]), get_t(w->attn_v_b[layer_idx]));

    tensor_t gather_q = m->tp_gather_q->slice(0, 0, seq_len);
    tensor_t gather_k = m->tp_gather_k->slice(0, 0, seq_len);
    tensor_t gather_v = m->tp_gather_v->slice(0, 0, seq_len);
    llaisysNcclAllGather(q_buf_local->data(), gather_q->data(), seq_len * nhdh_l, dtype, stream);
    llaisysNcclAllGather(k_buf_local->data(), gather_k->data(), seq_len * nkvhdh_l, dtype, stream);
    llaisysNcclAllGather(v_buf_local->data(), gather_v->data(), seq_len * nkvhdh_l, dtype, stream);
    llaisys::core::context().runtime().api()->stream_synchronize(stream);

    std::vector<size_t> shape_q = {seq_len, nh, dh};
    std::vector<size_t> shape_kv = {seq_len, nkvh, dh};
    tensor_t q_view = gather_q->view(shape_q);
    tensor_t k_view = gather_k->view(shape_kv);
    tensor_t v_view = gather_v->view(shape_kv);
    llaisys::ops::rope(q_rope, q_view, pos_ids_t, theta);
    llaisys::ops::rope(k_rope, k_view, pos_ids_t, theta);

    const size_t kv_row_bytes = nkvh * dh * elem_size;
    for (size_t s = 0; s < seq_len; s++) {
        size_t cache_pos = cache_start + s;
        copy_sync(
            reinterpret_cast<std::byte *>(k_cache->data()) + cache_pos * kv_row_bytes,
            reinterpret_cast<const std::byte *>(k_rope->data()) + s * kv_row_bytes,
            kv_row_bytes, m->device_type);
        copy_sync(
            reinterpret_cast<std::byte *>(v_cache->data()) + cache_pos * kv_row_bytes,
            reinterpret_cast<const std::byte *>(gather_v->data()) + s * nkvhdh * elem_size,
            kv_row_bytes, m->device_type);
    }
    size_t kv_len = cache_start + seq_len;
    tensor_t k_slice = k_cache->slice(0, 0, kv_len);
    tensor_t v_slice = v_cache->slice(0, 0, kv_len);
    llaisys::ops::self_attention(attn_val, q_rope, k_slice, v_slice, scale);

    std::vector<size_t> shape_attn_flat = {seq_len, nhdh};
    tensor_t attn_flat = attn_val->view(shape_attn_flat);
    llaisys::ops::linear(o_proj_out, attn_flat, get_t(w->attn_o_w[layer_idx]), nullptr);
    llaisysNcclAllReduce(o_proj_out->data(), o_proj_out->data(), seq_len * hs, dtype, stream);
    llaisys::core::context().runtime().api()->stream_synchronize(stream);
    llaisys::ops::add(res_buf, hidden, o_proj_out);
    copy_sync(hidden->data(), res_buf->data(), seq_len * hs * elem_size, m->device_type);

    llaisys::ops::rms_norm(normed, hidden, get_t(w->mlp_norm_w[layer_idx]), eps);
    llaisys::ops::linear(gate_buf_local, normed, get_t(w->mlp_gate_w[layer_idx]), nullptr);
    llaisys::ops::linear(up_buf_local, normed, get_t(w->mlp_up_w[layer_idx]), nullptr);
    tensor_t gather_gate = m->tp_gather_gate->slice(0, 0, seq_len);
    tensor_t gather_up = m->tp_gather_up->slice(0, 0, seq_len);
    llaisysNcclAllGather(gate_buf_local->data(), gather_gate->data(), seq_len * di_l, dtype, stream);
    llaisysNcclAllGather(up_buf_local->data(), gather_up->data(), seq_len * di_l, dtype, stream);
    llaisys::core::context().runtime().api()->stream_synchronize(stream);
    llaisys::ops::swiglu(mlp_buf, gather_gate, gather_up);
    llaisys::ops::linear(down_buf, mlp_buf, get_t(w->mlp_down_w[layer_idx]), nullptr);
    llaisysNcclAllReduce(down_buf->data(), down_buf->data(), seq_len * hs, dtype, stream);
    llaisys::core::context().runtime().api()->stream_synchronize(stream);
    llaisys::ops::add(res_buf, hidden, down_buf);
    copy_sync(hidden->data(), res_buf->data(), seq_len * hs * elem_size, m->device_type);
}
#endif

} // namespace

LLAISYS_EXTERN_C {

size_t llaisysQwen2ModelGetCacheLen(struct LlaisysQwen2Model *model) {
    return model && !model->cache_lens.empty() ? model->cache_lens[0] : 0;
}

size_t llaisysQwen2ModelGetCacheLenSlot(struct LlaisysQwen2Model *model, size_t slot_id) {
    if (!model || model->cache_lens.empty()) return 0;
    if (slot_id >= model->cache_lens.size()) return 0;
    return model->cache_lens[slot_id];
}

size_t llaisysQwen2ModelGetKVCacheBytes(struct LlaisysQwen2Model *model, size_t prefix_len) {
    if (!model || prefix_len == 0) return 0;
    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t elem_size = llaisys::utils::dsize(meta->dtype);
    return nlayer * 2 * prefix_len * nkvh * dh * elem_size;
}

} // extern "C"

// 内部辅助函数：使用 C++  linkage，避免 MSVC C4190（C-linkage 返回 C++ 类型）
static tensor_t get_slot_k_cache(LlaisysQwen2Model *model, size_t layer_idx, size_t slot_id) {
    const size_t maxseq = model->meta.maxseq, nkvh = model->meta.nkvh, dh = model->meta.dh;
    tensor_t raw = model->k_caches[layer_idx];
    if (model->meta.max_batch_size > 1)
        return raw->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
    return raw;
}
static tensor_t get_slot_v_cache(LlaisysQwen2Model *model, size_t layer_idx, size_t slot_id) {
    const size_t maxseq = model->meta.maxseq, nkvh = model->meta.nkvh, dh = model->meta.dh;
    tensor_t raw = model->v_caches[layer_idx];
    if (model->meta.max_batch_size > 1)
        return raw->slice(0, slot_id, slot_id + 1)->view({maxseq, nkvh, dh});
    return raw;
}

LLAISYS_EXTERN_C {

void llaisysQwen2ModelExportKVCache(struct LlaisysQwen2Model *model, void *ptr_out) {
    if (!model || !ptr_out || model->cache_lens.empty()) return;
    const size_t cache_len = model->cache_lens[0];
    if (cache_len == 0) return;
    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t elem_size = llaisys::utils::dsize(meta->dtype);
    const size_t row_bytes = nkvh * dh * elem_size;
    const size_t layer_bytes = cache_len * row_bytes;
    std::byte *out = static_cast<std::byte *>(ptr_out);
    llaisys::core::context().setDevice(model->device_type, model->device_id);
    for (size_t i = 0; i < nlayer; i++) {
        tensor_t k_slot = get_slot_k_cache(model, i, 0);
        tensor_t v_slot = get_slot_v_cache(model, i, 0);
        copy_device_to_host(out, k_slot->data(), layer_bytes, model->device_type, model->device_id);
        out += layer_bytes;
        copy_device_to_host(out, v_slot->data(), layer_bytes, model->device_type, model->device_id);
        out += layer_bytes;
    }
}

void llaisysQwen2ModelImportKVCache(struct LlaisysQwen2Model *model, const void *ptr_in, size_t prefix_len) {
    if (!model || !ptr_in || prefix_len == 0) return;
    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t elem_size = llaisys::utils::dsize(meta->dtype);
    const size_t row_bytes = nkvh * dh * elem_size;
    const size_t layer_bytes = prefix_len * row_bytes;
    const std::byte *in = static_cast<const std::byte *>(ptr_in);
    llaisys::core::context().setDevice(model->device_type, model->device_id);
    for (size_t i = 0; i < nlayer; i++) {
        tensor_t k_slot = get_slot_k_cache(model, i, 0);
        tensor_t v_slot = get_slot_v_cache(model, i, 0);
        copy_host_to_device(k_slot->data(), in, layer_bytes, model->device_type, model->device_id);
        in += layer_bytes;
        copy_host_to_device(v_slot->data(), in, layer_bytes, model->device_type, model->device_id);
        in += layer_bytes;
    }
    model->cache_lens[0] = prefix_len;
}

void llaisysQwen2ModelResetKVCache(struct LlaisysQwen2Model *model) {
    if (model) {
        for (size_t i = 0; i < model->cache_lens.size(); i++)
            model->cache_lens[i] = 0;
    }
}

void llaisysQwen2ModelResetKVCacheSlot(struct LlaisysQwen2Model *model, size_t slot_id) {
    if (!model || model->cache_lens.empty()) return;
    if (slot_id < model->cache_lens.size())
        model->cache_lens[slot_id] = 0;
}

void llaisysQwen2ModelExportKVCacheSlot(struct LlaisysQwen2Model *model, size_t slot_id, void *ptr_out) {
    if (!model || !ptr_out || model->cache_lens.empty()) return;
    if (slot_id >= model->cache_lens.size()) return;
    const size_t cache_len = model->cache_lens[slot_id];
    if (cache_len == 0) return;
    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t elem_size = llaisys::utils::dsize(meta->dtype);
    const size_t row_bytes = nkvh * dh * elem_size;
    const size_t layer_bytes = cache_len * row_bytes;
    std::byte *out = static_cast<std::byte *>(ptr_out);
    llaisys::core::context().setDevice(model->device_type, model->device_id);
    for (size_t i = 0; i < nlayer; i++) {
        tensor_t k_slot = get_slot_k_cache(model, i, slot_id);
        tensor_t v_slot = get_slot_v_cache(model, i, slot_id);
        copy_device_to_host(out, k_slot->data(), layer_bytes, model->device_type, model->device_id);
        out += layer_bytes;
        copy_device_to_host(out, v_slot->data(), layer_bytes, model->device_type, model->device_id);
        out += layer_bytes;
    }
}

void llaisysQwen2ModelImportKVCacheSlot(struct LlaisysQwen2Model *model, size_t slot_id, const void *ptr_in, size_t prefix_len) {
    if (!model || !ptr_in || prefix_len == 0 || model->cache_lens.empty()) return;
    if (slot_id >= model->cache_lens.size()) return;
    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t elem_size = llaisys::utils::dsize(meta->dtype);
    const size_t row_bytes = nkvh * dh * elem_size;
    const size_t layer_bytes = prefix_len * row_bytes;
    const std::byte *in = static_cast<const std::byte *>(ptr_in);
    llaisys::core::context().setDevice(model->device_type, model->device_id);
    for (size_t i = 0; i < nlayer; i++) {
        tensor_t k_slot = get_slot_k_cache(model, i, slot_id);
        tensor_t v_slot = get_slot_v_cache(model, i, slot_id);
        copy_host_to_device(k_slot->data(), in, layer_bytes, model->device_type, model->device_id);
        in += layer_bytes;
        copy_host_to_device(v_slot->data(), in, layer_bytes, model->device_type, model->device_id);
        in += layer_bytes;
    }
    model->cache_lens[slot_id] = prefix_len;
}

/**
 * 单步推理：根据当前 token 序列做一次前向，返回下一个 token 的 id。
 * - 若 cache_len==0（prefill）：传入整段 token_ids，seq_len=ntoken，并填充 KV cache；
 * - 若 cache_len>0 且 ntoken>1（suffix prefill）：传入后缀 token_ids，seq_len=ntoken，只对后缀做 prefill；
 * - 否则（decode）：只传入最后一个 token，seq_len=1，用已有 cache 做 attention。
 * - temperature<=0 或极小且 top_k<=1、top_p>=1 时使用 argmax；否则使用随机采样（Temperature/Top-K/Top-P）。
 */
int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model,
                                int64_t *token_ids,
                                size_t ntoken,
                                float temperature,
                                int top_k,
                                float top_p,
                                unsigned long long seed) {
    if (std::getenv("LLAISYS_DEBUG_INFER")) {
        std::fprintf(stderr, "[DBG] Infer entered model=%p ntoken=%zu\n",
                     (void *)model, (unsigned long)ntoken);
        std::fflush(stderr);
    }
    if (!model || ntoken == 0) return static_cast<int64_t>(-1);
    if (model->cache_lens.empty()) return static_cast<int64_t>(-1);

    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh, di = meta->di;
    const size_t voc = meta->voc;
    const llaisysDataType_t dtype = meta->dtype;
    const llaisysDeviceType_t dev = model->device_type;
    const int dev_id = model->device_id;

    const size_t slot0_len = model->cache_lens[0];
    const bool is_prefill = (slot0_len == 0);
    const bool is_suffix_prefill = (slot0_len > 0 && ntoken > 1);
    const size_t seq_len = is_suffix_prefill ? ntoken : (is_prefill ? ntoken : 1);
    const size_t cache_start = slot0_len;

    if (std::getenv("LLAISYS_DEBUG_INFER")) {
        std::fprintf(stderr, "[DBG] Infer entry ntoken=%zu seq_len=%zu prefill=%d\n",
                     (unsigned long)ntoken, (unsigned long)seq_len, is_prefill ? 1 : 0);
        std::fflush(stderr);
    }

    // GPU 且已全量缓存 CPU 权重：整次前向在 CPU 上执行。使用 k_caches_cpu 非空判断，避免仅 CacheOutputLayerOnCPU 时误入导致越界。
    if (dev != LLAISYS_DEVICE_CPU && !model->k_caches_cpu.empty()) {
        const bool use_sampling = (temperature > 1e-6f && (top_k > 1 || (top_p > 0.f && top_p < 1.f)));
        int64_t next_token = meta->end_token;
        llaisys::core::context().setDevice(LLAISYS_DEVICE_CPU, 0);

        tensor_t token_tensor = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        if (is_suffix_prefill)
            std::memcpy(token_tensor->data(), token_ids, ntoken * sizeof(int64_t));
        else if (is_prefill)
            std::memcpy(token_tensor->data(), token_ids, ntoken * sizeof(int64_t));
        else
            std::memcpy(token_tensor->data(), token_ids + ntoken - 1, sizeof(int64_t));

        tensor_t hidden = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t normed = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t q_buf = llaisys::Tensor::create({seq_len, nh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t k_buf = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t v_buf = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t q_rope = llaisys::Tensor::create({seq_len, nh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t k_rope = llaisys::Tensor::create({seq_len, nkvh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t attn_val = llaisys::Tensor::create({seq_len, nh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t o_proj_out = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t res_buf = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t gate_buf = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t up_buf = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t mlp_buf = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t down_buf = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t pos_ids_t = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        std::vector<int64_t> pos_ids_host(seq_len);
        for (size_t s = 0; s < seq_len; s++)
            pos_ids_host[s] = static_cast<int64_t>(cache_start + s);
        std::memcpy(pos_ids_t->data(), pos_ids_host.data(), seq_len * sizeof(int64_t));

        llaisys::ops::embedding(hidden, token_tensor, model->in_embed_cpu);
        for (size_t i = 0; i < nlayer; i++) {
            forward_layer_cpu(model, i, 0, hidden, normed, q_buf, k_buf, v_buf,
                              q_rope, k_rope, attn_val, o_proj_out, res_buf,
                              gate_buf, up_buf, mlp_buf, down_buf,
                              seq_len, cache_start, pos_ids_t);
        }
        model->cache_lens[0] = cache_start + seq_len;

        tensor_t normed_cpu = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::rms_norm(normed_cpu, hidden, model->out_norm_w_cpu, meta->epsilon);
        tensor_t logits_cpu = llaisys::Tensor::create({seq_len, voc}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::linear(logits_cpu, normed_cpu, model->out_embed_cpu, nullptr);
        tensor_t last_logit_1d = logits_cpu->slice(0, seq_len - 1, seq_len)->view(std::vector<size_t>{voc});

        if (use_sampling) {
            tensor_t sampled_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::sample(sampled_idx, last_logit_1d, temperature, top_k, top_p, static_cast<uint64_t>(seed));
            std::memcpy(&next_token, sampled_idx->data(), sizeof(int64_t));
        } else {
            tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::argmax(max_idx_t, max_val_t, last_logit_1d);
            std::memcpy(&next_token, max_idx_t->data(), sizeof(int64_t));
        }
        return next_token;
    }

    llaisys::core::context().setDevice(dev, dev_id);

    tensor_t token_tensor = llaisys::Tensor::create(
        {seq_len}, LLAISYS_DTYPE_I64, dev, dev_id);
    if (is_suffix_prefill) {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids, ntoken * sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    } else if (is_prefill) {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids, ntoken * sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    } else {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids + ntoken - 1, sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    }

    const bool use_tp_buf = (meta->tp_world_size > 1 && model->tp_gather_q != nullptr && dev == LLAISYS_DEVICE_NVIDIA);
    const size_t nhdh_buf = use_tp_buf ? (nh * dh / static_cast<size_t>(meta->tp_world_size)) : (nh * dh);
    const size_t nkvhdh_buf = use_tp_buf ? (nkvh * dh / static_cast<size_t>(meta->tp_world_size)) : (nkvh * dh);
    const size_t di_buf = use_tp_buf ? (di / static_cast<size_t>(meta->tp_world_size)) : di;

    tensor_t hidden = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t normed = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t q_buf = llaisys::Tensor::create({seq_len, nhdh_buf}, dtype, dev, dev_id);
    tensor_t k_buf = llaisys::Tensor::create({seq_len, nkvhdh_buf}, dtype, dev, dev_id);
    tensor_t v_buf = llaisys::Tensor::create({seq_len, nkvhdh_buf}, dtype, dev, dev_id);
    tensor_t q_rope = llaisys::Tensor::create({seq_len, nh, dh}, dtype, dev, dev_id);
    tensor_t k_rope = llaisys::Tensor::create({seq_len, nkvh, dh}, dtype, dev, dev_id);
    tensor_t attn_val = llaisys::Tensor::create({seq_len, nh, dh}, dtype, dev, dev_id);
    tensor_t o_proj_out = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t res_buf = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t gate_buf = llaisys::Tensor::create({seq_len, di_buf}, dtype, dev, dev_id);
    tensor_t up_buf = llaisys::Tensor::create({seq_len, di_buf}, dtype, dev, dev_id);
    tensor_t mlp_buf = llaisys::Tensor::create({seq_len, di}, dtype, dev, dev_id);
    tensor_t down_buf = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);

    tensor_t pos_ids_t = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, dev, dev_id);
    std::vector<int64_t> pos_ids_host(seq_len);
    for (size_t s = 0; s < seq_len; s++)
        pos_ids_host[s] = static_cast<int64_t>(cache_start + s);
    llaisys::core::context().runtime().api()->memcpy_sync(
        pos_ids_t->data(), pos_ids_host.data(), seq_len * sizeof(int64_t),
        (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);

    if (dev != LLAISYS_DEVICE_CPU && model->in_embed_cpu) {
        tensor_t token_cpu = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        if (is_suffix_prefill) std::memcpy(token_cpu->data(), token_ids, ntoken * sizeof(int64_t));
        else if (is_prefill) std::memcpy(token_cpu->data(), token_ids, ntoken * sizeof(int64_t));
        else std::memcpy(token_cpu->data(), token_ids + ntoken - 1, sizeof(int64_t));
        tensor_t hidden_cpu = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::embedding(hidden_cpu, token_cpu, model->in_embed_cpu);
        llaisys::core::context().setDevice(dev, dev_id);
        copy_host_to_device(hidden->data(), hidden_cpu->data(), seq_len * hs * llaisys::utils::dsize(dtype), dev, dev_id);
        llaisys::core::context().setDevice(dev, dev_id);
    } else {
        llaisys::ops::embedding(hidden, token_tensor, get_t(model->weights.in_embed));
    }

    const bool use_tp_infer = (meta->tp_world_size > 1 && model->tp_gather_q != nullptr && dev == LLAISYS_DEVICE_NVIDIA);
    if (use_tp_infer) {
#ifdef ENABLE_NCCL
        void *stream = llaisys::core::context().runtime().stream();
        for (size_t i = 0; i < nlayer; i++) {
            forward_layer_tp(model, i, 0, hidden, normed, q_buf, k_buf, v_buf,
                             q_rope, k_rope, attn_val, o_proj_out, res_buf,
                             gate_buf, up_buf, mlp_buf, down_buf,
                             seq_len, cache_start, pos_ids_t, stream);
        }
#else
        (void)seq_len;
        (void)cache_start;
        for (size_t i = 0; i < nlayer; i++) {
            forward_layer(model, i, 0, hidden, normed, q_buf, k_buf, v_buf,
                         q_rope, k_rope, attn_val, o_proj_out, res_buf,
                         gate_buf, up_buf, mlp_buf, down_buf,
                         seq_len, cache_start, pos_ids_t);
        }
#endif
    } else {
        for (size_t i = 0; i < nlayer; i++) {
            forward_layer(model, i, 0, hidden, normed, q_buf, k_buf, v_buf,
                         q_rope, k_rope, attn_val, o_proj_out, res_buf,
                         gate_buf, up_buf, mlp_buf, down_buf,
                         seq_len, cache_start, pos_ids_t);
        }
    }
    model->cache_lens[0] = cache_start + seq_len;

    const bool use_sampling = (temperature > 1e-6f && (top_k > 1 || (top_p > 0.f && top_p < 1.f)));
    int64_t next_token = meta->end_token;
    const size_t elem_size = llaisys::utils::dsize(dtype);

    // GPU 且已缓存输出层 CPU 权重时：最后一层 norm + linear 在 CPU 上算，规避 GPU 输出层异常
    if (dev != LLAISYS_DEVICE_CPU && model->out_norm_w_cpu && model->out_embed_cpu) {
        if (std::getenv("LLAISYS_DEBUG_INFER")) { std::fprintf(stderr, "[DBG] Infer before out_cpu sync+d2h\n"); std::fflush(stderr); }
        llaisys::core::context().setDevice(dev, dev_id);
        llaisys::core::context().runtime().api()->device_synchronize();
        // 整块拷贝 hidden 到 CPU，再在 CPU 上取最后一行，避免 D2H 按“最后一行”拷贝时 stride/layout 与 GPU 不一致
        const size_t full_bytes = seq_len * hs * elem_size;
        tensor_t hidden_cpu_full = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(hidden_cpu_full->data(), hidden->data(), full_bytes, dev, dev_id);
        tensor_t hidden_last = hidden_cpu_full->slice(0, seq_len - 1, seq_len);
        if (std::getenv("LLAISYS_DEBUG_INFER")) { std::fprintf(stderr, "[DBG] Infer after d2h\n"); std::fflush(stderr); }
        tensor_t normed_cpu = llaisys::Tensor::create({1, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::rms_norm(normed_cpu, hidden_last, model->out_norm_w_cpu, meta->epsilon);
        tensor_t logits_cpu = llaisys::Tensor::create({1, voc}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::linear(logits_cpu, normed_cpu, model->out_embed_cpu, nullptr);
        tensor_t last_logit_1d = logits_cpu->view(std::vector<size_t>{voc});

        if (use_sampling) {
            tensor_t sampled_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::sample(sampled_idx, last_logit_1d, temperature, top_k, top_p, static_cast<uint64_t>(seed));
            llaisys::core::context().runtime().api()->memcpy_sync(
                &next_token, sampled_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
        } else {
            tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::argmax(max_idx_t, max_val_t, last_logit_1d);
            llaisys::core::context().runtime().api()->memcpy_sync(
                &next_token, max_idx_t->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
        }
        if (std::getenv("LLAISYS_DEBUG_INFER")) {
            const std::byte *logit_ptr = last_logit_1d->data();
            auto logit_at = [&](size_t i) -> float {
                if (i >= voc) return 0.f;
                const std::byte *p = logit_ptr + i * elem_size;
                switch (dtype) {
                case LLAISYS_DTYPE_F32: return *reinterpret_cast<const float *>(p);
                case LLAISYS_DTYPE_F16: return llaisys::utils::cast<float>(*reinterpret_cast<const llaisys::fp16_t *>(p));
                case LLAISYS_DTYPE_BF16: return llaisys::utils::cast<float>(*reinterpret_cast<const llaisys::bf16_t *>(p));
                default: return 0.f;
                }
            };
            std::fprintf(stderr, "[DBG] Infer return(out_cpu) next=%lld logit[0]=%.4f logit[15]=%.4f logit[%lld]=%.4f\n",
                (long long)next_token, logit_at(0), logit_at(15), (long long)next_token, logit_at(static_cast<size_t>(next_token)));
            std::fflush(stderr);
        }
        return next_token;
    }

    // 默认路径：在当前设备执行输出层的 RMSNorm 与 Linear 投影
    llaisys::ops::rms_norm(normed, hidden, get_t(model->weights.out_norm_w), meta->epsilon);

    tensor_t logits_t = llaisys::Tensor::create({seq_len, voc}, dtype, dev, dev_id);
    llaisys::ops::linear(logits_t, normed, get_t(model->weights.out_embed), nullptr);

    tensor_t last_logit = logits_t->slice(0, seq_len - 1, seq_len);
    tensor_t last_logit_1d = last_logit->view(std::vector<size_t>{voc});

    if (use_sampling) {
        tensor_t logits_for_sample = last_logit_1d;
        if (dev != LLAISYS_DEVICE_CPU) {
            llaisys::core::context().runtime().api()->device_synchronize();
            tensor_t logits_cpu = llaisys::Tensor::create({voc}, dtype, LLAISYS_DEVICE_CPU, 0);
            const size_t logits_row_bytes = voc * elem_size;
            const std::byte *src_row = reinterpret_cast<const std::byte *>(logits_t->data()) + (seq_len - 1) * logits_row_bytes;
            llaisys::core::context().runtime().api()->memcpy_sync(
                logits_cpu->data(), src_row, logits_row_bytes, LLAISYS_MEMCPY_D2H);
            logits_for_sample = logits_cpu;
        }
        tensor_t sampled_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::sample(sampled_idx, logits_for_sample, temperature, top_k, top_p, static_cast<uint64_t>(seed));
        llaisys::core::context().runtime().api()->memcpy_sync(
            &next_token, sampled_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
    } else {
        if (dev != LLAISYS_DEVICE_CPU) {
            llaisys::core::context().runtime().api()->device_synchronize();
            tensor_t logits_cpu = llaisys::Tensor::create({voc}, dtype, LLAISYS_DEVICE_CPU, 0);
            const std::byte *src_row = reinterpret_cast<const std::byte *>(logits_t->data()) + (seq_len - 1) * voc * elem_size;
            llaisys::core::context().runtime().api()->memcpy_sync(
                logits_cpu->data(), src_row, voc * elem_size, LLAISYS_MEMCPY_D2H);
            tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::argmax(max_idx_t, max_val_t, logits_cpu->view(std::vector<size_t>{voc}));
            llaisys::core::context().runtime().api()->memcpy_sync(
                &next_token, max_idx_t->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
        } else {
            tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, dev, dev_id);
            tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, dev, dev_id);
            llaisys::ops::argmax(max_idx_t, max_val_t, last_logit_1d);
            llaisys::core::context().runtime().api()->memcpy_sync(
                &next_token, max_idx_t->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
        }
    }

    return next_token;
}

int64_t llaisysQwen2ModelInferHybrid(struct LlaisysQwen2Model *model,
                                      int64_t *token_ids,
                                      size_t ntoken,
                                      float temperature,
                                      int top_k,
                                      float top_p,
                                      unsigned long long seed,
                                      int gpu_up_to_layer) {
    if (!model || ntoken == 0) return static_cast<int64_t>(-1);
    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh, di = meta->di;
    const size_t voc = meta->voc;
    const llaisysDataType_t dtype = meta->dtype;
    const llaisysDeviceType_t dev = model->device_type;
    const int dev_id = model->device_id;

    const size_t slot0_len = model->cache_lens[0];
    const bool is_prefill = (slot0_len == 0);
    const bool is_suffix_prefill = (slot0_len > 0 && ntoken > 1);
    const size_t seq_len = is_suffix_prefill ? ntoken : (is_prefill ? ntoken : 1);
    const size_t cache_start = slot0_len;

    if (!model->in_embed_cpu) return llaisysQwen2ModelInfer(model, token_ids, ntoken, temperature, top_k, top_p, seed);
    // 仅 embedding+输出层在 CPU（未调用 CacheAllWeightsOnCPU）时不能走全量 CPU 分支，否则 forward_layer_cpu 会访问空的 k_caches_cpu
    if (model->k_caches_cpu.empty()) return llaisysQwen2ModelInfer(model, token_ids, ntoken, temperature, top_k, top_p, seed);

    const bool use_sampling = (temperature > 1e-6f && (top_k > 1 || (top_p > 0.f && top_p < 1.f)));
    int64_t next_token = meta->end_token;
    const size_t elem_size = llaisys::utils::dsize(dtype);

    if (gpu_up_to_layer < 0) {
        llaisys::core::context().setDevice(LLAISYS_DEVICE_CPU, 0);
        tensor_t token_tensor = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        if (is_suffix_prefill) std::memcpy(token_tensor->data(), token_ids, ntoken * sizeof(int64_t));
        else if (is_prefill) std::memcpy(token_tensor->data(), token_ids, ntoken * sizeof(int64_t));
        else std::memcpy(token_tensor->data(), token_ids + ntoken - 1, sizeof(int64_t));
        tensor_t hidden = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t normed = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t q_buf = llaisys::Tensor::create({seq_len, nh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t k_buf = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t v_buf = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t q_rope = llaisys::Tensor::create({seq_len, nh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t k_rope = llaisys::Tensor::create({seq_len, nkvh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t attn_val = llaisys::Tensor::create({seq_len, nh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t o_proj_out = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t res_buf = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t gate_buf = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t up_buf = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t mlp_buf = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t down_buf = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t pos_ids_t = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        std::vector<int64_t> pos_ids_host(seq_len);
        for (size_t s = 0; s < seq_len; s++) pos_ids_host[s] = static_cast<int64_t>(cache_start + s);
        std::memcpy(pos_ids_t->data(), pos_ids_host.data(), seq_len * sizeof(int64_t));
        llaisys::ops::embedding(hidden, token_tensor, model->in_embed_cpu);
        for (size_t i = 0; i < nlayer; i++)
            forward_layer_cpu(model, i, 0, hidden, normed, q_buf, k_buf, v_buf, q_rope, k_rope, attn_val, o_proj_out, res_buf, gate_buf, up_buf, mlp_buf, down_buf, seq_len, cache_start, pos_ids_t);
        model->cache_lens[0] = cache_start + seq_len;
        tensor_t normed_cpu = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::rms_norm(normed_cpu, hidden, model->out_norm_w_cpu, meta->epsilon);
        tensor_t logits_cpu = llaisys::Tensor::create({seq_len, voc}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::linear(logits_cpu, normed_cpu, model->out_embed_cpu, nullptr);
        tensor_t last_logit_1d = logits_cpu->slice(0, seq_len - 1, seq_len)->view(std::vector<size_t>{voc});
        if (use_sampling) {
            tensor_t sampled_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::sample(sampled_idx, last_logit_1d, temperature, top_k, top_p, static_cast<uint64_t>(seed));
            std::memcpy(&next_token, sampled_idx->data(), sizeof(int64_t));
        } else {
            tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::argmax(max_idx_t, max_val_t, last_logit_1d);
            std::memcpy(&next_token, max_idx_t->data(), sizeof(int64_t));
        }
        return next_token;
    }

    int gpu_layers = static_cast<int>(gpu_up_to_layer);
    if (gpu_layers >= static_cast<int>(nlayer)) gpu_layers = static_cast<int>(nlayer) - 1;

    llaisys::core::context().setDevice(dev, dev_id);
    tensor_t token_tensor = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, dev, dev_id);
    if (is_suffix_prefill) llaisys::core::context().runtime().api()->memcpy_sync(token_tensor->data(), token_ids, ntoken * sizeof(int64_t), LLAISYS_MEMCPY_H2D);
    else if (is_prefill) llaisys::core::context().runtime().api()->memcpy_sync(token_tensor->data(), token_ids, ntoken * sizeof(int64_t), LLAISYS_MEMCPY_H2D);
    else llaisys::core::context().runtime().api()->memcpy_sync(token_tensor->data(), token_ids + ntoken - 1, sizeof(int64_t), LLAISYS_MEMCPY_H2D);

    const bool use_tp_buf = (meta->tp_world_size > 1 && model->tp_gather_q != nullptr && dev == LLAISYS_DEVICE_NVIDIA);
    const size_t nhdh_buf = use_tp_buf ? (nh * dh / static_cast<size_t>(meta->tp_world_size)) : (nh * dh);
    const size_t nkvhdh_buf = use_tp_buf ? (nkvh * dh / static_cast<size_t>(meta->tp_world_size)) : (nkvh * dh);
    const size_t di_buf = use_tp_buf ? (di / static_cast<size_t>(meta->tp_world_size)) : di;

    tensor_t hidden = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t normed = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t q_buf = llaisys::Tensor::create({seq_len, nhdh_buf}, dtype, dev, dev_id);
    tensor_t k_buf = llaisys::Tensor::create({seq_len, nkvhdh_buf}, dtype, dev, dev_id);
    tensor_t v_buf = llaisys::Tensor::create({seq_len, nkvhdh_buf}, dtype, dev, dev_id);
    tensor_t q_rope = llaisys::Tensor::create({seq_len, nh, dh}, dtype, dev, dev_id);
    tensor_t k_rope = llaisys::Tensor::create({seq_len, nkvh, dh}, dtype, dev, dev_id);
    tensor_t attn_val = llaisys::Tensor::create({seq_len, nh, dh}, dtype, dev, dev_id);
    tensor_t o_proj_out = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t res_buf = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t gate_buf = llaisys::Tensor::create({seq_len, di_buf}, dtype, dev, dev_id);
    tensor_t up_buf = llaisys::Tensor::create({seq_len, di_buf}, dtype, dev, dev_id);
    tensor_t mlp_buf = llaisys::Tensor::create({seq_len, di}, dtype, dev, dev_id);
    tensor_t down_buf = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t pos_ids_t = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, dev, dev_id);
    std::vector<int64_t> pos_ids_host(seq_len);
    for (size_t s = 0; s < seq_len; s++) pos_ids_host[s] = static_cast<int64_t>(cache_start + s);
    llaisys::core::context().runtime().api()->memcpy_sync(pos_ids_t->data(), pos_ids_host.data(), seq_len * sizeof(int64_t), LLAISYS_MEMCPY_H2D);

    llaisys::ops::embedding(hidden, token_tensor, get_t(model->weights.in_embed));
    for (int i = 0; i <= gpu_layers; i++)
        forward_layer(model, i, 0, hidden, normed, q_buf, k_buf, v_buf, q_rope, k_rope, attn_val, o_proj_out, res_buf, gate_buf, up_buf, mlp_buf, down_buf, seq_len, cache_start, pos_ids_t);

    llaisys::core::context().runtime().api()->device_synchronize();
    tensor_t hidden_cpu = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
    copy_device_to_host(hidden_cpu->data(), hidden->data(), seq_len * hs * elem_size, dev, dev_id);

    llaisys::core::context().setDevice(LLAISYS_DEVICE_CPU, 0);
    tensor_t normed_cpu = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t q_buf_cpu = llaisys::Tensor::create({seq_len, nh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t k_buf_cpu = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t v_buf_cpu = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t q_rope_cpu = llaisys::Tensor::create({seq_len, nh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t k_rope_cpu = llaisys::Tensor::create({seq_len, nkvh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t attn_val_cpu = llaisys::Tensor::create({seq_len, nh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t o_proj_out_cpu = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t res_buf_cpu = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t gate_buf_cpu = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t up_buf_cpu = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t mlp_buf_cpu = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t down_buf_cpu = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
    tensor_t pos_ids_cpu = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    std::memcpy(pos_ids_cpu->data(), pos_ids_host.data(), seq_len * sizeof(int64_t));

    for (size_t i = static_cast<size_t>(gpu_layers) + 1; i < nlayer; i++)
        forward_layer_cpu(model, i, 0, hidden_cpu, normed_cpu, q_buf_cpu, k_buf_cpu, v_buf_cpu, q_rope_cpu, k_rope_cpu, attn_val_cpu, o_proj_out_cpu, res_buf_cpu, gate_buf_cpu, up_buf_cpu, mlp_buf_cpu, down_buf_cpu, seq_len, cache_start, pos_ids_cpu);

    model->cache_lens[0] = cache_start + seq_len;
    tensor_t normed_out = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
    llaisys::ops::rms_norm(normed_out, hidden_cpu, model->out_norm_w_cpu, meta->epsilon);
    tensor_t logits_cpu = llaisys::Tensor::create({seq_len, voc}, dtype, LLAISYS_DEVICE_CPU, 0);
    llaisys::ops::linear(logits_cpu, normed_out, model->out_embed_cpu, nullptr);
    tensor_t last_logit_1d = logits_cpu->slice(0, seq_len - 1, seq_len)->view(std::vector<size_t>{voc});
    if (use_sampling) {
        tensor_t sampled_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::sample(sampled_idx, last_logit_1d, temperature, top_k, top_p, static_cast<uint64_t>(seed));
        std::memcpy(&next_token, sampled_idx->data(), sizeof(int64_t));
    } else {
        tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::argmax(max_idx_t, max_val_t, last_logit_1d);
        std::memcpy(&next_token, max_idx_t->data(), sizeof(int64_t));
    }
    return next_token;
}

int64_t llaisysQwen2ModelInferWithSlot(struct LlaisysQwen2Model *model,
                                        size_t slot_id,
                                        int64_t *token_ids,
                                        size_t ntoken,
                                        float temperature,
                                        int top_k,
                                        float top_p,
                                        unsigned long long seed) {
    if (!model || ntoken == 0) return static_cast<int64_t>(-1);
    if (slot_id >= model->cache_lens.size()) return static_cast<int64_t>(-1);

    const LlaisysQwen2Meta *meta = &model->meta;
    const size_t nlayer = meta->nlayer;
    const size_t hs = meta->hs, nh = meta->nh, nkvh = meta->nkvh, dh = meta->dh, di = meta->di;
    const size_t voc = meta->voc;
    const llaisysDataType_t dtype = meta->dtype;
    const llaisysDeviceType_t dev = model->device_type;
    const int dev_id = model->device_id;

    size_t *p_cache_len = &model->cache_lens[slot_id];
    const bool is_prefill = (*p_cache_len == 0);
    const bool is_suffix_prefill = (*p_cache_len > 0 && ntoken > 1);
    const size_t seq_len = is_suffix_prefill ? ntoken : (is_prefill ? ntoken : 1);
    const size_t cache_start = *p_cache_len;

    // GPU 且已全量缓存 CPU 权重：整次前向在 CPU 上执行。使用 k_caches_cpu 非空判断，避免仅 CacheOutputLayerOnCPU 时误入导致越界。
    if (dev != LLAISYS_DEVICE_CPU && !model->k_caches_cpu.empty()) {
        const bool use_sampling = (temperature > 1e-6f && (top_k > 1 || (top_p > 0.f && top_p < 1.f)));
        int64_t next_token = meta->end_token;
        llaisys::core::context().setDevice(LLAISYS_DEVICE_CPU, 0);

        tensor_t token_tensor = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        if (is_suffix_prefill)
            std::memcpy(token_tensor->data(), token_ids, ntoken * sizeof(int64_t));
        else if (is_prefill)
            std::memcpy(token_tensor->data(), token_ids, ntoken * sizeof(int64_t));
        else
            std::memcpy(token_tensor->data(), token_ids + ntoken - 1, sizeof(int64_t));

        tensor_t hidden = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t normed = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t q_buf = llaisys::Tensor::create({seq_len, nh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t k_buf = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t v_buf = llaisys::Tensor::create({seq_len, nkvh * dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t q_rope = llaisys::Tensor::create({seq_len, nh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t k_rope = llaisys::Tensor::create({seq_len, nkvh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t attn_val = llaisys::Tensor::create({seq_len, nh, dh}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t o_proj_out = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t res_buf = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t gate_buf = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t up_buf = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t mlp_buf = llaisys::Tensor::create({seq_len, di}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t down_buf = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        tensor_t pos_ids_t = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        std::vector<int64_t> pos_ids_host(seq_len);
        for (size_t s = 0; s < seq_len; s++)
            pos_ids_host[s] = static_cast<int64_t>(cache_start + s);
        std::memcpy(pos_ids_t->data(), pos_ids_host.data(), seq_len * sizeof(int64_t));

        llaisys::ops::embedding(hidden, token_tensor, model->in_embed_cpu);
        for (size_t i = 0; i < nlayer; i++) {
            forward_layer_cpu(model, i, slot_id, hidden, normed, q_buf, k_buf, v_buf,
                             q_rope, k_rope, attn_val, o_proj_out, res_buf,
                             gate_buf, up_buf, mlp_buf, down_buf,
                             seq_len, cache_start, pos_ids_t);
        }
        *p_cache_len = cache_start + seq_len;

        tensor_t normed_cpu = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::rms_norm(normed_cpu, hidden, model->out_norm_w_cpu, meta->epsilon);
        tensor_t logits_cpu = llaisys::Tensor::create({seq_len, voc}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::linear(logits_cpu, normed_cpu, model->out_embed_cpu, nullptr);
        tensor_t last_logit_1d = logits_cpu->slice(0, seq_len - 1, seq_len)->view(std::vector<size_t>{voc});

        if (use_sampling) {
            tensor_t sampled_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::sample(sampled_idx, last_logit_1d, temperature, top_k, top_p, static_cast<uint64_t>(seed));
            std::memcpy(&next_token, sampled_idx->data(), sizeof(int64_t));
        } else {
            tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::argmax(max_idx_t, max_val_t, last_logit_1d);
            std::memcpy(&next_token, max_idx_t->data(), sizeof(int64_t));
        }
        return next_token;
    }

    llaisys::core::context().setDevice(dev, dev_id);

    tensor_t token_tensor = llaisys::Tensor::create(
        {seq_len}, LLAISYS_DTYPE_I64, dev, dev_id);
    if (is_suffix_prefill) {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids, ntoken * sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    } else if (is_prefill) {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids, ntoken * sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    } else {
        llaisys::core::context().runtime().api()->memcpy_sync(
            token_tensor->data(), token_ids + ntoken - 1, sizeof(int64_t),
            (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
    }

    const bool use_tp_buf = (meta->tp_world_size > 1 && model->tp_gather_q != nullptr && dev == LLAISYS_DEVICE_NVIDIA);
    const size_t nhdh_buf = use_tp_buf ? (nh * dh / static_cast<size_t>(meta->tp_world_size)) : (nh * dh);
    const size_t nkvhdh_buf = use_tp_buf ? (nkvh * dh / static_cast<size_t>(meta->tp_world_size)) : (nkvh * dh);
    const size_t di_buf = use_tp_buf ? (di / static_cast<size_t>(meta->tp_world_size)) : di;

    tensor_t hidden = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t normed = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t q_buf = llaisys::Tensor::create({seq_len, nhdh_buf}, dtype, dev, dev_id);
    tensor_t k_buf = llaisys::Tensor::create({seq_len, nkvhdh_buf}, dtype, dev, dev_id);
    tensor_t v_buf = llaisys::Tensor::create({seq_len, nkvhdh_buf}, dtype, dev, dev_id);
    tensor_t q_rope = llaisys::Tensor::create({seq_len, nh, dh}, dtype, dev, dev_id);
    tensor_t k_rope = llaisys::Tensor::create({seq_len, nkvh, dh}, dtype, dev, dev_id);
    tensor_t attn_val = llaisys::Tensor::create({seq_len, nh, dh}, dtype, dev, dev_id);
    tensor_t o_proj_out = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t res_buf = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);
    tensor_t gate_buf = llaisys::Tensor::create({seq_len, di_buf}, dtype, dev, dev_id);
    tensor_t up_buf = llaisys::Tensor::create({seq_len, di_buf}, dtype, dev, dev_id);
    tensor_t mlp_buf = llaisys::Tensor::create({seq_len, di}, dtype, dev, dev_id);
    tensor_t down_buf = llaisys::Tensor::create({seq_len, hs}, dtype, dev, dev_id);

    tensor_t pos_ids_t = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, dev, dev_id);
    std::vector<int64_t> pos_ids_host(seq_len);
    for (size_t s = 0; s < seq_len; s++)
        pos_ids_host[s] = static_cast<int64_t>(cache_start + s);
    llaisys::core::context().runtime().api()->memcpy_sync(
        pos_ids_t->data(), pos_ids_host.data(), seq_len * sizeof(int64_t),
        (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);

    if (dev != LLAISYS_DEVICE_CPU && model->in_embed_cpu) {
        tensor_t token_cpu = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        if (is_suffix_prefill) std::memcpy(token_cpu->data(), token_ids, ntoken * sizeof(int64_t));
        else if (is_prefill) std::memcpy(token_cpu->data(), token_ids, ntoken * sizeof(int64_t));
        else std::memcpy(token_cpu->data(), token_ids + ntoken - 1, sizeof(int64_t));
        tensor_t hidden_cpu = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::embedding(hidden_cpu, token_cpu, model->in_embed_cpu);
        llaisys::core::context().setDevice(dev, dev_id);
        copy_host_to_device(hidden->data(), hidden_cpu->data(), seq_len * hs * llaisys::utils::dsize(dtype), dev, dev_id);
        llaisys::core::context().setDevice(dev, dev_id);
    } else {
        llaisys::ops::embedding(hidden, token_tensor, get_t(model->weights.in_embed));
    }

    const bool use_tp_slot = (meta->tp_world_size > 1 && model->tp_gather_q != nullptr && dev == LLAISYS_DEVICE_NVIDIA);
    if (use_tp_slot) {
#ifdef ENABLE_NCCL
        void *stream = llaisys::core::context().runtime().stream();
        for (size_t i = 0; i < nlayer; i++) {
            forward_layer_tp(model, i, slot_id, hidden, normed, q_buf, k_buf, v_buf,
                            q_rope, k_rope, attn_val, o_proj_out, res_buf,
                            gate_buf, up_buf, mlp_buf, down_buf,
                            seq_len, cache_start, pos_ids_t, stream);
        }
#else
        for (size_t i = 0; i < nlayer; i++) {
            forward_layer(model, i, slot_id, hidden, normed, q_buf, k_buf, v_buf,
                         q_rope, k_rope, attn_val, o_proj_out, res_buf,
                         gate_buf, up_buf, mlp_buf, down_buf,
                         seq_len, cache_start, pos_ids_t);
        }
#endif
    } else {
        for (size_t i = 0; i < nlayer; i++) {
            forward_layer(model, i, slot_id, hidden, normed, q_buf, k_buf, v_buf,
                         q_rope, k_rope, attn_val, o_proj_out, res_buf,
                         gate_buf, up_buf, mlp_buf, down_buf,
                         seq_len, cache_start, pos_ids_t);
        }
    }

    *p_cache_len = cache_start + seq_len;

    const bool use_sampling = (temperature > 1e-6f && (top_k > 1 || (top_p > 0.f && top_p < 1.f)));
    int64_t next_token = meta->end_token;
    const size_t elem_size = llaisys::utils::dsize(dtype);

    if (dev != LLAISYS_DEVICE_CPU && model->out_norm_w_cpu && model->out_embed_cpu) {
        llaisys::core::context().setDevice(dev, dev_id);
        llaisys::core::context().runtime().api()->device_synchronize();
        // 整块拷贝 hidden 到 CPU，再在 CPU 上取最后一行
        const size_t full_bytes = seq_len * hs * elem_size;
        tensor_t hidden_cpu_full = llaisys::Tensor::create({seq_len, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        copy_device_to_host(hidden_cpu_full->data(), hidden->data(), full_bytes, dev, dev_id);
        tensor_t hidden_last = hidden_cpu_full->slice(0, seq_len - 1, seq_len);
        tensor_t normed_cpu = llaisys::Tensor::create({1, hs}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::rms_norm(normed_cpu, hidden_last, model->out_norm_w_cpu, meta->epsilon);
        tensor_t logits_cpu = llaisys::Tensor::create({1, voc}, dtype, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::linear(logits_cpu, normed_cpu, model->out_embed_cpu, nullptr);
        tensor_t last_logit_1d = logits_cpu->view(std::vector<size_t>{voc});

        if (use_sampling) {
            tensor_t sampled_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::sample(sampled_idx, last_logit_1d, temperature, top_k, top_p, static_cast<uint64_t>(seed));
            llaisys::core::context().runtime().api()->memcpy_sync(
                &next_token, sampled_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
        } else {
            tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::argmax(max_idx_t, max_val_t, last_logit_1d);
            llaisys::core::context().runtime().api()->memcpy_sync(
                &next_token, max_idx_t->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
        }
        if (std::getenv("LLAISYS_DEBUG_INFER")) {
            const std::byte *logit_ptr = last_logit_1d->data();
            auto logit_at = [&](size_t i) -> float {
                if (i >= voc) return 0.f;
                const std::byte *p = logit_ptr + i * elem_size;
                switch (dtype) {
                case LLAISYS_DTYPE_F32: return *reinterpret_cast<const float *>(p);
                case LLAISYS_DTYPE_F16: return llaisys::utils::cast<float>(*reinterpret_cast<const llaisys::fp16_t *>(p));
                case LLAISYS_DTYPE_BF16: return llaisys::utils::cast<float>(*reinterpret_cast<const llaisys::bf16_t *>(p));
                default: return 0.f;
                }
            };
            std::fprintf(stderr, "[DBG] Infer return(out_cpu slot) next=%lld logit[0]=%.4f logit[15]=%.4f logit[%lld]=%.4f\n",
                (long long)next_token, logit_at(0), logit_at(15), (long long)next_token, logit_at(static_cast<size_t>(next_token)));
            std::fflush(stderr);
        }
        return next_token;
    }

    // 默认路径：在当前设备执行输出层的 RMSNorm 与 Linear 投影
    llaisys::ops::rms_norm(normed, hidden, get_t(model->weights.out_norm_w), meta->epsilon);

    tensor_t logits_t = llaisys::Tensor::create({seq_len, voc}, dtype, dev, dev_id);
    llaisys::ops::linear(logits_t, normed, get_t(model->weights.out_embed), nullptr);

    tensor_t last_logit = logits_t->slice(0, seq_len - 1, seq_len);
    tensor_t last_logit_1d = last_logit->view(std::vector<size_t>{voc});

    if (use_sampling) {
        tensor_t logits_for_sample = last_logit_1d;
        if (dev != LLAISYS_DEVICE_CPU) {
            llaisys::core::context().runtime().api()->device_synchronize();
            tensor_t logits_cpu = llaisys::Tensor::create({voc}, dtype, LLAISYS_DEVICE_CPU, 0);
            const std::byte *src_row = reinterpret_cast<const std::byte *>(logits_t->data()) + (seq_len - 1) * voc * elem_size;
            llaisys::core::context().runtime().api()->memcpy_sync(
                logits_cpu->data(), src_row, voc * elem_size, LLAISYS_MEMCPY_D2H);
            logits_for_sample = logits_cpu;
        }
        tensor_t sampled_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        llaisys::ops::sample(sampled_idx, logits_for_sample, temperature, top_k, top_p, static_cast<uint64_t>(seed));
        llaisys::core::context().runtime().api()->memcpy_sync(
            &next_token, sampled_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
    } else {
        if (dev != LLAISYS_DEVICE_CPU) {
            llaisys::core::context().runtime().api()->device_synchronize();
            tensor_t logits_cpu = llaisys::Tensor::create({voc}, dtype, LLAISYS_DEVICE_CPU, 0);
            const std::byte *src_row = reinterpret_cast<const std::byte *>(logits_t->data()) + (seq_len - 1) * voc * elem_size;
            llaisys::core::context().runtime().api()->memcpy_sync(
                logits_cpu->data(), src_row, voc * elem_size, LLAISYS_MEMCPY_D2H);
            tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, LLAISYS_DEVICE_CPU, 0);
            llaisys::ops::argmax(max_idx_t, max_val_t, logits_cpu->view(std::vector<size_t>{voc}));
            llaisys::core::context().runtime().api()->memcpy_sync(
                &next_token, max_idx_t->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
        } else {
            tensor_t max_idx_t = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, dev, dev_id);
            tensor_t max_val_t = llaisys::Tensor::create({1}, dtype, dev, dev_id);
            llaisys::ops::argmax(max_idx_t, max_val_t, last_logit_1d);
            llaisys::core::context().runtime().api()->memcpy_sync(
                &next_token, max_idx_t->data(), sizeof(int64_t), LLAISYS_MEMCPY_H2H);
        }
    }

    return next_token;
}

void llaisysQwen2ModelBatchedDecode(struct LlaisysQwen2Model *model,
                                     const size_t *slot_ids,
                                     const int64_t *token_ids,
                                     size_t n_batch,
                                     int64_t *out_next_tokens,
                                     float temperature,
                                     int top_k,
                                     float top_p,
                                     unsigned long long seed) {
    if (!model || !slot_ids || !token_ids || !out_next_tokens || n_batch == 0)
        return;
    const size_t max_batch = model->meta.max_batch_size;
    if (n_batch > max_batch)
        n_batch = max_batch;
    for (size_t i = 0; i < n_batch; i++) {
        int64_t one_token = token_ids[i];
        out_next_tokens[i] = llaisysQwen2ModelInferWithSlot(
            model, slot_ids[i], &one_token, 1,
            temperature, top_k, top_p, seed);
    }
}

} // extern "C"
