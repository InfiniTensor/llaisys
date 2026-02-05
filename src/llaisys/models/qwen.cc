#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

#include "llaisys/models/qwen2.h"
// #include "tensor/tensor.hpp"
// #include "llaisys/ops.hpp"
// #include "llaisys/utils/check.hpp"
// #include "llaisys/core/context.hpp"
#include "../../tensor/tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils/check.hpp"
#include "../../core/context/context.hpp"
#include "../llaisys_tensor.hpp"



// ===== Debug helpers =====
#ifndef LLAISYS_QWEN2_DEBUG
#define LLAISYS_QWEN2_DEBUG 0
#endif

#if LLAISYS_QWEN2_DEBUG
#include <iostream>
#define DBG_MSG(x) do { std::cout << "[DBG] " << x << std::endl; } while (0)
#define DBG_TENSOR(name, t) \
    do { \
        std::cout << "\n[DBG_TENSOR] " << (name) << std::endl; \
        (t)->debug(); \
    } while (0)
#else
#define DBG_MSG(x) do {} while (0)
#define DBG_TENSOR(name, t) do {} while (0)
#endif



namespace {

using llaisys::tensor_t;

static inline tensor_t make_tensor(const std::vector<size_t>& shape,
                                   llaisysDataType_t dtype,
                                   llaisysDeviceType_t dev,
                                   int dev_id) {
    return llaisys::Tensor::create(shape, dtype, dev, dev_id);
}

// 强健：确保传给 ops 的张量是 contiguous（因为 ops 不看 strides）
static inline void require_contiguous(const tensor_t& t, const char* name) {
    ASSERT(t->isContiguous(), name);
}

struct Qwen2KV {
    tensor_t k; // [maxseq, nkvh, dh]
    tensor_t v; // [maxseq, nkvh, dh]
};

struct Qwen2Tmp {
    tensor_t tok_i64;   // [max_L]
    tensor_t pos_i64;   // [max_L]

    // residual stream（我们用 swap 避免 add 后 memcpy）
    tensor_t x_a;       // [max_L, hs]
    tensor_t x_b;       // [max_L, hs] scratch

    tensor_t h;         // [max_L, hs]
    tensor_t y;         // [max_L, hs]

    tensor_t q1, k1, v1;    // [max_L, nh*dh] / [max_L, nkvh*dh]
    tensor_t q, k, v;       // [max_L, nh, dh] / [max_L, nkvh, dh]
    tensor_t q_rope, k_rope;

    tensor_t attn_val;      // [max_L, nh, dh]
    tensor_t attn_merge;    // [max_L, nh*dh]
    tensor_t attn_out;      // [max_L, hs]

    tensor_t mlp_in;        // [max_L, hs]
    tensor_t gate, up, act; // [max_L, di]
    tensor_t mlp_out;       // [max_L, hs]

    tensor_t logits;        // [1, voc]
    tensor_t max_idx;       // [1] i64
    tensor_t max_val;       // [1] dt
};

struct Qwen2Impl {
    LlaisysQwen2Meta meta{};
    llaisysDeviceType_t device{};
    int device_id{0};

    LlaisysQwen2Weights weights{};

    std::vector<Qwen2KV> kv;
    Qwen2Tmp tmp{};

    size_t cur_pos{0};
};

static void alloc_weights(Qwen2Impl* m) {
    auto& meta = m->meta;
    auto& w = m->weights;

    const size_t nlayer = meta.nlayer;
    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;

    const auto dt = meta.dtype;
    const auto dev = m->device;
    const int dev_id = m->device_id;

    w.in_embed   = new LlaisysTensor{ make_tensor({voc, hs}, dt, dev, dev_id) };
    w.out_embed  = new LlaisysTensor{ make_tensor({voc, hs}, dt, dev, dev_id) };
    w.out_norm_w = new LlaisysTensor{ make_tensor({hs}, dt, dev, dev_id) };

    w.attn_norm_w = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));
    w.attn_q_w    = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));
    w.attn_q_b    = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));
    w.attn_k_w    = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));
    w.attn_k_b    = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));
    w.attn_v_w    = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));
    w.attn_v_b    = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));
    w.attn_o_w    = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));

    w.mlp_norm_w  = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));
    w.mlp_gate_w  = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));
    w.mlp_up_w    = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));
    w.mlp_down_w  = (llaisysTensor_t*)std::calloc(nlayer, sizeof(llaisysTensor_t));

    for (size_t i = 0; i < nlayer; ++i) {
        w.attn_norm_w[i] = new LlaisysTensor{ make_tensor({hs}, meta.dtype, dev, dev_id) };

        w.attn_q_w[i] = new LlaisysTensor{ make_tensor({nh*dh, hs}, meta.dtype, dev, dev_id) };
        w.attn_q_b[i] = new LlaisysTensor{ make_tensor({nh*dh},     meta.dtype, dev, dev_id) };

        w.attn_k_w[i] = new LlaisysTensor{ make_tensor({nkvh*dh, hs}, meta.dtype, dev, dev_id) };
        w.attn_k_b[i] = new LlaisysTensor{ make_tensor({nkvh*dh},     meta.dtype, dev, dev_id) };

        w.attn_v_w[i] = new LlaisysTensor{ make_tensor({nkvh*dh, hs}, meta.dtype, dev, dev_id) };
        w.attn_v_b[i] = new LlaisysTensor{ make_tensor({nkvh*dh},     meta.dtype, dev, dev_id) };

        w.attn_o_w[i] = new LlaisysTensor{ make_tensor({hs, nh*dh}, meta.dtype, dev, dev_id) };

        w.mlp_norm_w[i] = new LlaisysTensor{ make_tensor({hs}, meta.dtype, dev, dev_id) };
        w.mlp_gate_w[i] = new LlaisysTensor{ make_tensor({di, hs}, meta.dtype, dev, dev_id) };
        w.mlp_up_w[i]   = new LlaisysTensor{ make_tensor({di, hs}, meta.dtype, dev, dev_id) };
        w.mlp_down_w[i] = new LlaisysTensor{ make_tensor({hs, di}, meta.dtype, dev, dev_id) };
    }
}

static void alloc_kv_tmp(Qwen2Impl* m, size_t max_L) {
    auto& meta = m->meta;
    const auto dt = meta.dtype;
    const auto dev = m->device;
    const int dev_id = m->device_id;

    const size_t nlayer = meta.nlayer;
    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;

    // KV cache：用模型 dtype（与你 ops 一致）
    m->kv.resize(nlayer);
    for (size_t i = 0; i < nlayer; ++i) {
        m->kv[i].k = make_tensor({meta.maxseq, nkvh, dh}, dt, dev, dev_id);
        m->kv[i].v = make_tensor({meta.maxseq, nkvh, dh}, dt, dev, dev_id);
        std::memset(m->kv[i].k->data(), 0, m->kv[i].k->numel() * m->kv[i].k->elementSize());
        std::memset(m->kv[i].v->data(), 0, m->kv[i].v->numel() * m->kv[i].v->elementSize());
    }

    auto& t = m->tmp;
    t.tok_i64 = make_tensor({max_L}, LLAISYS_DTYPE_I64, dev, dev_id);
    t.pos_i64 = make_tensor({max_L}, LLAISYS_DTYPE_I64, dev, dev_id);

    t.x_a = make_tensor({max_L, hs}, dt, dev, dev_id);
    t.x_b = make_tensor({max_L, hs}, dt, dev, dev_id);

    t.h   = make_tensor({max_L, hs}, dt, dev, dev_id);
    t.y   = make_tensor({max_L, hs}, dt, dev, dev_id);

    t.q1  = make_tensor({max_L, nh*dh},   dt, dev, dev_id);
    t.k1  = make_tensor({max_L, nkvh*dh}, dt, dev, dev_id);
    t.v1  = make_tensor({max_L, nkvh*dh}, dt, dev, dev_id);

    t.q      = make_tensor({max_L, nh, dh},   dt, dev, dev_id);
    t.k      = make_tensor({max_L, nkvh, dh}, dt, dev, dev_id);
    t.v      = make_tensor({max_L, nkvh, dh}, dt, dev, dev_id);
    t.q_rope = make_tensor({max_L, nh, dh},   dt, dev, dev_id);
    t.k_rope = make_tensor({max_L, nkvh, dh}, dt, dev, dev_id);

    t.attn_val   = make_tensor({max_L, nh, dh}, dt, dev, dev_id);
    t.attn_merge = make_tensor({max_L, nh*dh},  dt, dev, dev_id);
    t.attn_out   = make_tensor({max_L, hs},     dt, dev, dev_id);

    t.mlp_in  = make_tensor({max_L, hs}, dt, dev, dev_id);
    t.gate    = make_tensor({max_L, di}, dt, dev, dev_id);
    t.up      = make_tensor({max_L, di}, dt, dev, dev_id);
    t.act     = make_tensor({max_L, di}, dt, dev, dev_id);
    t.mlp_out = make_tensor({max_L, hs}, dt, dev, dev_id);

    t.logits  = make_tensor({1, voc}, dt, dev, dev_id);
    t.max_idx = make_tensor({1}, LLAISYS_DTYPE_I64, dev, dev_id);
    t.max_val = make_tensor({1}, dt, dev, dev_id);
}

static void free_weights(Qwen2Impl* m) {
    auto& w = m->weights;
    auto del_t = [](llaisysTensor_t t){ if (t) delete t; };

    del_t(w.in_embed);
    del_t(w.out_embed);
    del_t(w.out_norm_w);

    for (size_t i = 0; i < m->meta.nlayer; ++i) {
        del_t(w.attn_norm_w[i]);
        del_t(w.attn_q_w[i]);
        del_t(w.attn_q_b[i]);
        del_t(w.attn_k_w[i]);
        del_t(w.attn_k_b[i]);
        del_t(w.attn_v_w[i]);
        del_t(w.attn_v_b[i]);
        del_t(w.attn_o_w[i]);

        del_t(w.mlp_norm_w[i]);
        del_t(w.mlp_gate_w[i]);
        del_t(w.mlp_up_w[i]);
        del_t(w.mlp_down_w[i]);
    }

    std::free(w.attn_norm_w);
    std::free(w.attn_q_w);
    std::free(w.attn_q_b);
    std::free(w.attn_k_w);
    std::free(w.attn_k_b);
    std::free(w.attn_v_w);
    std::free(w.attn_v_b);
    std::free(w.attn_o_w);

    std::free(w.mlp_norm_w);
    std::free(w.mlp_gate_w);
    std::free(w.mlp_up_w);
    std::free(w.mlp_down_w);
}

} // namespace

extern "C" {

struct LlaisysQwen2Model {
    Qwen2Impl* impl;
};

__export struct LlaisysQwen2Model *
llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                        llaisysDeviceType_t device,
                        int* /*device_ids*/, int /*ndevice*/) noexcept　{
    CHECK_ARGUMENT(meta != nullptr, "meta is null");

    auto* model = new LlaisysQwen2Model;
    model->impl = new Qwen2Impl;
    auto* m = model->impl;

    m->meta = *meta;
    m->device = device;
    m->device_id = 0;
    m->cur_pos = 0;

    llaisys::core::context().setDevice(device, 0);

    alloc_weights(m);

    // 保险：tmp 直接按 maxseq 分配，避免 prompt 超出临时 buffer
    alloc_kv_tmp(m, meta->maxseq);

    return model;
}

__export void
llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) noexcept　{
    if (!model) return;
    if (model->impl) {
        free_weights(model->impl);
        delete model->impl;
    }
    delete model;
}

__export struct LlaisysQwen2Weights *
llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) noexcept {
    return model ? &model->impl->weights : nullptr;
}

__export int64_t
llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) noexcept {
    CHECK_ARGUMENT(model && model->impl, "model is null");
    CHECK_ARGUMENT(token_ids != nullptr, "token_ids is null");
    CHECK_ARGUMENT(ntoken > 0, "ntoken must be > 0");

    auto* m = model->impl;
    auto& meta = m->meta;
    auto& w = m->weights;
    auto& t = m->tmp;

    CHECK_ARGUMENT(m->cur_pos + ntoken < meta.maxseq, "sequence length exceeds maxseq");

    const size_t dh = meta.dh;
    const float scale = 1.0f / std::sqrt((float)dh);

    // ---- 写 tok/pos ----
    std::memcpy(t.tok_i64->data(), token_ids, ntoken * sizeof(int64_t));
    {
        int64_t* p = reinterpret_cast<int64_t*>(t.pos_i64->data());
        for (size_t i = 0; i < ntoken; ++i) p[i] = (int64_t)(m->cur_pos + i);
    }

    // 只做 dim0 slice：通常仍 contiguous
    tensor_t tok_L = t.tok_i64->slice(0, 0, ntoken);
    tensor_t pos_L = t.pos_i64->slice(0, 0, ntoken);

    tensor_t x_L  = t.x_a->slice(0, 0, ntoken);
    tensor_t xb_L = t.x_b->slice(0, 0, ntoken);
    tensor_t h_L  = t.h->slice(0, 0, ntoken);

    require_contiguous(tok_L, "tok_L must be contiguous");
    require_contiguous(pos_L, "pos_L must be contiguous");
    require_contiguous(x_L,   "x_L must be contiguous");

    // ---- embedding ----
    llaisys::ops::embedding(x_L, tok_L, w.in_embed->tensor);

    // after embedding(x_L, tok_L, in_embed)
    if (LLAISYS_QWEN2_DEBUG) {
        auto x0 = x_L->slice(0, 0, 1);  // [1, hs]
        DBG_TENSOR("x after embedding (first token)", x0);
    }

    // ---- layers ----
    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        // attn rmsnorm
        llaisys::ops::rms_norm(h_L, x_L, w.attn_norm_w[layer]->tensor, meta.epsilon);

        // qkv
        tensor_t q1_L = t.q1->slice(0, 0, ntoken);
        tensor_t k1_L = t.k1->slice(0, 0, ntoken);
        tensor_t v1_L = t.v1->slice(0, 0, ntoken);

        llaisys::ops::linear(q1_L, h_L, w.attn_q_w[layer]->tensor, w.attn_q_b[layer]->tensor);
        llaisys::ops::linear(k1_L, h_L, w.attn_k_w[layer]->tensor, w.attn_k_b[layer]->tensor);
        llaisys::ops::linear(v1_L, h_L, w.attn_v_w[layer]->tensor, w.attn_v_b[layer]->tensor);

        // rearrange to [L, head, dh] （你们是 memcpy，所以 numel 必须相同）
        tensor_t q_L = t.q->slice(0, 0, ntoken);
        tensor_t k_L = t.k->slice(0, 0, ntoken);
        tensor_t v_L = t.v->slice(0, 0, ntoken);

        llaisys::ops::rearrange(q_L, q1_L);
        llaisys::ops::rearrange(k_L, k1_L);
        llaisys::ops::rearrange(v_L, v1_L);

        // rope
        tensor_t q_rope_L = t.q_rope->slice(0, 0, ntoken);
        tensor_t k_rope_L = t.k_rope->slice(0, 0, ntoken);
        llaisys::ops::rope(q_rope_L, q_L, pos_L, meta.theta);
        llaisys::ops::rope(k_rope_L, k_L, pos_L, meta.theta);

        // after rope(q_rope_L, q_L, pos_L, theta) and rope(k_rope_L, ...)
        if (LLAISYS_QWEN2_DEBUG && layer == 0) {
            auto q0 = q_rope_L->slice(0, 0, 1); // [1, nh, dh]
            auto k0 = k_rope_L->slice(0, 0, 1); // [1, nkvh, dh]
            DBG_TENSOR("layer0 q_rope (first token)", q0);
            DBG_TENSOR("layer0 k_rope (first token)", k0);
        }


        // write KV cache
        tensor_t kc_win = m->kv[layer].k->slice(0, m->cur_pos, m->cur_pos + ntoken);
        tensor_t vc_win = m->kv[layer].v->slice(0, m->cur_pos, m->cur_pos + ntoken);
        // 注意：slice dim0，连续 OK
        llaisys::ops::rearrange(kc_win, k_rope_L);
        llaisys::ops::rearrange(vc_win, v_L);

        // history KV for attention
        const size_t total_len = m->cur_pos + ntoken;
        tensor_t k_hist = m->kv[layer].k->slice(0, 0, total_len);
        tensor_t v_hist = m->kv[layer].v->slice(0, 0, total_len);

        tensor_t attn_val_L = t.attn_val->slice(0, 0, ntoken);
        llaisys::ops::self_attention(attn_val_L, q_rope_L, k_hist, v_hist, scale);

        // merge heads
        tensor_t attn_merge_L = t.attn_merge->slice(0, 0, ntoken);
        llaisys::ops::rearrange(attn_merge_L, attn_val_L);

        // o proj
        tensor_t attn_out_L = t.attn_out->slice(0, 0, ntoken);
        llaisys::ops::linear(attn_out_L, attn_merge_L, w.attn_o_w[layer]->tensor, nullptr);

        // residual: xb = x + attn_out; swap(x, xb)
        llaisys::ops::add(xb_L, x_L, attn_out_L);
        std::swap(t.x_a, t.x_b);
        x_L  = t.x_a->slice(0, 0, ntoken);
        xb_L = t.x_b->slice(0, 0, ntoken);

        // MLP
        tensor_t mlp_in_L = t.mlp_in->slice(0, 0, ntoken);
        llaisys::ops::rms_norm(mlp_in_L, x_L, w.mlp_norm_w[layer]->tensor, meta.epsilon);

        tensor_t gate_L = t.gate->slice(0, 0, ntoken);
        tensor_t up_L   = t.up->slice(0, 0, ntoken);
        llaisys::ops::linear(gate_L, mlp_in_L, w.mlp_gate_w[layer]->tensor, nullptr);
        llaisys::ops::linear(up_L,   mlp_in_L, w.mlp_up_w[layer]->tensor,   nullptr);

        tensor_t act_L = t.act->slice(0, 0, ntoken);
        llaisys::ops::swiglu(act_L, gate_L, up_L);

        tensor_t mlp_out_L = t.mlp_out->slice(0, 0, ntoken);
        llaisys::ops::linear(mlp_out_L, act_L, w.mlp_down_w[layer]->tensor, nullptr);

        // residual: xb = x + mlp_out; swap
        llaisys::ops::add(xb_L, x_L, mlp_out_L);
        std::swap(t.x_a, t.x_b);
        x_L  = t.x_a->slice(0, 0, ntoken);
        xb_L = t.x_b->slice(0, 0, ntoken);
    }

    // final norm
    tensor_t y_L = t.y->slice(0, 0, ntoken);
    llaisys::ops::rms_norm(y_L, x_L, w.out_norm_w->tensor, meta.epsilon);

    // last token hidden: [1, hs]
    tensor_t y_last = y_L->slice(0, ntoken - 1, ntoken);

    // logits: [1, voc]
    llaisys::ops::linear(t.logits, y_last, w.out_embed->tensor, nullptr);
    if (LLAISYS_QWEN2_DEBUG) {
        DBG_TENSOR("logits (shape [1, voc])", t.logits);
    }

    // argmax over logits numel (你们 argmax 扫全 numel)
    llaisys::ops::argmax(t.max_idx, t.max_val, t.logits);

    int64_t next = *reinterpret_cast<const int64_t*>(t.max_idx->data());

    m->cur_pos += ntoken;
    return next;
}

} // extern "C"
