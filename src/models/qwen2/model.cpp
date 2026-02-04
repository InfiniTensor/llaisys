#include "model.hpp"

#include "../../llaisys/llaisys_tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../tensor/tensor.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace llaisys::models {

static inline llaisys::tensor_t unwrap_(llaisysTensor_t h, const char *what) {
    CHECK_ARGUMENT(h != nullptr, what);
    return h->tensor;
}

static inline void ensure_kv_cache_allocated_(std::vector<llaisys::tensor_t> &k_cache,
                                              std::vector<llaisys::tensor_t> &v_cache,
                                              const LlaisysQwen2Meta &meta,
                                              llaisysDeviceType_t device_type,
                                              int device_id) {
    const bool already_allocated = (k_cache.size() == meta.nlayer && v_cache.size() == meta.nlayer && meta.nlayer > 0
                                    && k_cache[0] != nullptr && v_cache[0] != nullptr);
    if (already_allocated) {
        return;
    }

    k_cache.assign(meta.nlayer, nullptr);
    v_cache.assign(meta.nlayer, nullptr);
    for (size_t layer = 0; layer < meta.nlayer; layer++) {
        k_cache[layer] = llaisys::Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
        v_cache[layer] = llaisys::Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
    }
}

static inline void kv_cache_write_row_(llaisys::tensor_t cache,
                                       llaisys::tensor_t kv_new,
                                       size_t pos,
                                       size_t nkvh,
                                       size_t dh) {
    CHECK_ARGUMENT(cache != nullptr && kv_new != nullptr, "kv_cache_write_row: null tensor");
    CHECK_ARGUMENT(cache->isContiguous() && kv_new->isContiguous(), "kv_cache_write_row: tensors must be contiguous");
    CHECK_ARGUMENT(cache->ndim() == 3 && kv_new->ndim() == 3, "kv_cache_write_row: expected 3D tensors");
    CHECK_ARGUMENT(kv_new->shape()[0] == 1 && kv_new->shape()[1] == nkvh && kv_new->shape()[2] == dh,
                   "kv_cache_write_row: kv_new must be [1, nkvh, dh]");
    CHECK_ARGUMENT(cache->shape()[0] > pos && cache->shape()[1] == nkvh && cache->shape()[2] == dh,
                   "kv_cache_write_row: cache must be [maxseq, nkvh, dh] with pos in range");

    const size_t row_elems = nkvh * dh;
    const size_t row_bytes = row_elems * cache->elementSize();
    std::memcpy(cache->data() + pos * row_bytes, kv_new->data(), row_bytes);
}

std::unique_ptr<Qwen2Model> Qwen2Model::create(const LlaisysQwen2Meta &meta,
                                               llaisysDeviceType_t device,
                                               const int *device_ids,
                                               int ndevice) {
    CHECK_ARGUMENT(meta.nlayer > 0, "Qwen2Model: meta.nlayer must be > 0");
    CHECK_ARGUMENT(meta.hs > 0 && meta.nh > 0 && meta.dh > 0, "Qwen2Model: invalid hidden/head sizes");
    CHECK_ARGUMENT(meta.nh % meta.nkvh == 0, "Qwen2Model: require nh % nkvh == 0");
    CHECK_ARGUMENT(meta.maxseq > 0 && meta.voc > 0, "Qwen2Model: invalid maxseq/vocab");

    std::vector<int> ids;
    if (device_ids && ndevice > 0) {
        ids.assign(device_ids, device_ids + ndevice);
    } else {
        ids.push_back(0);
    }

    return std::unique_ptr<Qwen2Model>(new Qwen2Model(meta, device, std::move(ids)));
}

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta,
                       llaisysDeviceType_t device,
                       std::vector<int> device_ids)
    : meta_(meta), device_(device), device_ids_(std::move(device_ids)) {
    // Allocate per-layer handle arrays (to be populated from Python).
    attn_norm_w_.assign(meta_.nlayer, nullptr);
    attn_q_w_.assign(meta_.nlayer, nullptr);
    attn_q_b_.assign(meta_.nlayer, nullptr);
    attn_k_w_.assign(meta_.nlayer, nullptr);
    attn_k_b_.assign(meta_.nlayer, nullptr);
    attn_v_w_.assign(meta_.nlayer, nullptr);
    attn_v_b_.assign(meta_.nlayer, nullptr);
    attn_o_w_.assign(meta_.nlayer, nullptr);

    mlp_norm_w_.assign(meta_.nlayer, nullptr);
    mlp_gate_w_.assign(meta_.nlayer, nullptr);
    mlp_up_w_.assign(meta_.nlayer, nullptr);
    mlp_down_w_.assign(meta_.nlayer, nullptr);

    k_cache_.assign(meta_.nlayer, nullptr);
    v_cache_.assign(meta_.nlayer, nullptr);
    cur_len_ = 0;
    initWeightsView_();
}

Qwen2Model::~Qwen2Model() {
}

void Qwen2Model::initWeightsView_() {
    std::memset(&weights_, 0, sizeof(weights_));
    // Global weights (to be filled by Python)
    weights_.in_embed = nullptr;
    weights_.out_embed = nullptr;
    weights_.out_norm_w = nullptr;

    // Per-layer arrays (stable pointers to vector storage).
    weights_.attn_norm_w = attn_norm_w_.data();
    weights_.attn_q_w = attn_q_w_.data();
    weights_.attn_q_b = attn_q_b_.data();
    weights_.attn_k_w = attn_k_w_.data();
    weights_.attn_k_b = attn_k_b_.data();
    weights_.attn_v_w = attn_v_w_.data();
    weights_.attn_v_b = attn_v_b_.data();
    weights_.attn_o_w = attn_o_w_.data();

    weights_.mlp_norm_w = mlp_norm_w_.data();
    weights_.mlp_gate_w = mlp_gate_w_.data();
    weights_.mlp_up_w = mlp_up_w_.data();
    weights_.mlp_down_w = mlp_down_w_.data();
}

LlaisysQwen2Weights *Qwen2Model::weights() {
    return &weights_;
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr || ntoken == 0, "Qwen2ModelInfer: token_ids is null");

    // Lesson 5 skeleton (scaffold, not full math):
    //
    // 0) Validate required weights are loaded (fail fast).
    // 1) Token -> embedding
    //    - gather token embedding(s) from weights_.in_embed
    // 2) For each layer i in [0, nlayer)
    //    2.1) Attn norm (RMSNorm): x = rms_norm(x, attn_norm_w[i])
    //    2.2) QKV projections: q = xWq + bq, k = xWk + bk, v = xWv + bv
    //    2.3) RoPE on q/k with position ids
    //    2.4) Self-attention with causal mask + KV-cache
    //    2.5) Output projection: attn_out = attn_val @ Wo
    //    2.6) Residual add: x = x + attn_out
    //
    //    2.7) MLP norm (RMSNorm): x = rms_norm(x, mlp_norm_w[i])
    //    2.8) MLP projections: gate = xW_gate, up = xW_up
    //    2.9) SwiGLU: act = swiglu(gate, up)
    //    2.10) Down proj: mlp_out = act @ W_down
    //    2.11) Residual add: x = x + mlp_out
    //
    // 3) Final norm: x = rms_norm(x, out_norm_w)
    // 4) LM head: logits = x @ out_embed^T
    // 5) Argmax (for --test): next_token = argmax(logits)
    //
    // KV-cache: allocate per-layer K/V buffers sized [maxseq, nkvh, dh] and update at each step.

    const char *hint = missingWeightsHint_();
    CHECK_ARGUMENT(hint == nullptr, hint);

    // Minimal "first real computation" skeleton:
    // - last_token_id -> embedding -> final_norm -> lm_head -> argmax
    // This is NOT a correct model forward yet (layers + KV-cache still TODO), but it validates
    // that weights are usable and ops glue works.
    CHECK_ARGUMENT(device_ == LLAISYS_DEVICE_CPU, "Qwen2Model: only CPU infer is wired in the skeleton");

    if (ntoken == 0) {
        return meta_.end_token;
    }
    CHECK_ARGUMENT(ntoken <= meta_.maxseq, "Qwen2ModelInfer: ntoken exceeds maxseq");
    if (0 == cur_len_) {
        // prefill
        ensure_kv_cache_allocated_(k_cache_, v_cache_, meta_, device_, 0);

        std::vector<int64_t> vec_id(ntoken);
        for (size_t i = 0; i < ntoken; ++i) {
            CHECK_ARGUMENT(token_ids[i] >= 0 && static_cast<size_t>(token_ids[i]) < meta_.voc, "Qwen2ModelInfer: token id out of range");
            vec_id[i] = token_ids[i];
        }

        // index: [T] i64
        auto index = llaisys::Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device_, 0);
        index->load(vec_id.data());

        // x: [T, H]
        auto x = llaisys::Tensor::create({ntoken, meta_.hs}, meta_.dtype, device_, 0);
        llaisys::ops::embedding(x, index, unwrap_(weights_.in_embed, "Qwen2Model: weights.in_embed is null"));

        // pos_ids: [T] i64
        std::vector<int64_t> vec_pos(ntoken);
        for (size_t i = 0; i < ntoken; ++i) {
            vec_pos[i] = i;
        }
        auto pos_ids = llaisys::Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device_, 0);
        pos_ids->load(vec_pos.data());
        for (size_t layer = 0; layer < meta_.nlayer; ++layer) {
            CHECK_ARGUMENT(k_cache_[layer] != nullptr && v_cache_[layer] != nullptr, "Qwen2ModelInfer(prefill): kv cache is null");
            CHECK_ARGUMENT(k_cache_[layer]->ndim() == 3 && v_cache_[layer]->ndim() == 3, "Qwen2ModelInfer(prefill): kv cache must be 3D");
            CHECK_ARGUMENT(k_cache_[layer]->shape()[0] == meta_.maxseq && v_cache_[layer]->shape()[0] == meta_.maxseq,
                           "Qwen2ModelInfer(prefill): kv cache maxseq mismatch");
            CHECK_ARGUMENT(k_cache_[layer]->shape()[1] == meta_.nkvh && v_cache_[layer]->shape()[1] == meta_.nkvh,
                           "Qwen2ModelInfer(prefill): kv cache nkvh mismatch");
            CHECK_ARGUMENT(k_cache_[layer]->shape()[2] == meta_.dh && v_cache_[layer]->shape()[2] == meta_.dh,
                           "Qwen2ModelInfer(prefill): kv cache dh mismatch");

            // attn_norm:[T,H]
            auto x_attn_in = llaisys::Tensor::create({ntoken, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::rms_norm(x_attn_in, x, unwrap_(weights_.attn_norm_w[layer], "Qwen2Model: weights.attn_norm_w is null"), meta_.epsilon);

            // q2d:[T,H]
            auto q2d = llaisys::Tensor::create({ntoken, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::linear(
                q2d,
                x_attn_in,
                unwrap_(weights_.attn_q_w[layer], "Qwen2Model: weights.attn_q_w is null"),
                unwrap_(weights_.attn_q_b[layer], "Qwen2Model: weights.attn_q_b is null"));

            // k2d:[T,nkvh*hd]
            auto k2d = llaisys::Tensor::create({ntoken, meta_.nkvh * meta_.dh}, meta_.dtype, device_, 0);
            llaisys::ops::linear(
                k2d,
                x_attn_in,
                unwrap_(weights_.attn_k_w[layer], "Qwen2Model: weights.attn_k_w is null"),
                unwrap_(weights_.attn_k_b[layer], "Qwen2Model: weights.attn_k_b is null"));

            // v2d:[T,nkvh*hd]
            auto v2d = llaisys::Tensor::create({ntoken, meta_.nkvh * meta_.dh}, meta_.dtype, device_, 0);
            llaisys::ops::linear(
                v2d,
                x_attn_in,
                unwrap_(weights_.attn_v_w[layer], "Qwen2Model: weights.attn_v_w is null"),
                unwrap_(weights_.attn_v_b[layer], "Qwen2Model: weights.attn_v_b is null"));

            // q:[T,nh,nd]
            auto q = q2d->view({ntoken, meta_.nh, meta_.dh});

            // k:[T,nkvh,nd]
            auto k = k2d->view({ntoken, meta_.nkvh, meta_.dh});

            // v:[T,nkvh,nd]
            auto v = v2d->view({ntoken, meta_.nkvh, meta_.dh});
            for (size_t i = 0; i < ntoken; i++) {
                auto slice = v->slice(0, i, i + 1);
                kv_cache_write_row_(v_cache_[layer], slice, i, meta_.nkvh, meta_.dh);
            }
            // q_rope [T, nh, hd]
            auto q_rope = llaisys::Tensor::create({ntoken, meta_.nh, meta_.dh}, meta_.dtype, device_, 0);
            llaisys::ops::rope(q_rope, q, pos_ids, meta_.theta);

            // k_rope [T, nkvh, hd]
            auto k_rope = llaisys::Tensor::create({ntoken, meta_.nkvh, meta_.dh}, meta_.dtype, device_, 0);
            llaisys::ops::rope(k_rope, k, pos_ids, meta_.theta);
            for (size_t i = 0; i < ntoken; i++) {
                auto slice = k_rope->slice(0, i, i + 1);
                kv_cache_write_row_(k_cache_[layer], slice, i, meta_.nkvh, meta_.dh);
            }

            // attn_val [T,nh,hd]
            auto attn_val = llaisys::Tensor::create({ntoken, meta_.nh, meta_.dh}, meta_.dtype, device_, 0);
            llaisys::ops::self_attention(attn_val, q_rope, k_rope, v, float(1.0f / std::sqrt(meta_.dh)));

            // attn2d [T,H]
            auto attn2d = attn_val->view({ntoken, meta_.hs});

            // attn_out [T,H]
            auto attn_out = llaisys::Tensor::create({ntoken, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::linear(attn_out, attn2d, unwrap_(weights_.attn_o_w[layer], "Qwen2Model: weights.attn_o_ws is null"), nullptr);

            // x_after_attn [T,H]
            auto x_after_attn = llaisys::Tensor::create({ntoken, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::add(x_after_attn, x, attn_out);

            // x_mlp_in [T,H]
            auto x_mlp_in = llaisys::Tensor::create({ntoken, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::rms_norm(x_mlp_in, x_after_attn, unwrap_(weights_.mlp_norm_w[layer], "Qwen2Model: weights.mlp_norm_w is null"), meta_.epsilon);

            // gate [T,di]
            auto gate = llaisys::Tensor::create({ntoken, meta_.di}, meta_.dtype, device_, 0);
            llaisys::ops::linear(gate, x_mlp_in, unwrap_(weights_.mlp_gate_w[layer], "Qwen2Model: weights_.mlp_gate_w is null"), nullptr);

            // up [T,di]
            auto up = llaisys::Tensor::create({ntoken, meta_.di}, meta_.dtype, device_, 0);
            llaisys::ops::linear(up, x_mlp_in, unwrap_(weights_.mlp_up_w[layer], "Qwen2Model: weights_.mlp_up_w is null"), nullptr);

            // act [T,di]
            auto act = llaisys::Tensor::create({ntoken, meta_.di}, meta_.dtype, device_, 0);
            llaisys::ops::swiglu(act, gate, up);

            // mlp_out [T,H]
            auto mlp_out = llaisys::Tensor::create({ntoken, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::linear(mlp_out, act, unwrap_(weights_.mlp_down_w[layer], "Qwen2Model: weights_.mlp_down_w is null"), nullptr);

            // x_after_mlp [T,H]
            llaisys::ops::add(x, x_after_attn, mlp_out);
        }

        // final norm: [T, H]
        auto last_x = x->slice(0, ntoken - 1, ntoken);
        auto x_norm = llaisys::Tensor::create({1, meta_.hs}, meta_.dtype, device_, 0);
        llaisys::ops::rms_norm(x_norm, last_x, unwrap_(weights_.out_norm_w, "Qwen2Model: weights.out_norm_w is null"), meta_.epsilon);

        // logits2d: [1, vocab]
        auto logits2d = llaisys::Tensor::create({1, meta_.voc}, meta_.dtype, device_, 0);
        llaisys::ops::linear(
            logits2d,
            x_norm,
            unwrap_(weights_.out_embed, "Qwen2Model: weights.out_embed is null"),
            nullptr);

        // argmax over vocab (treat logits as 1D [vocab])
        auto logits = logits2d->view({meta_.voc});
        auto max_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device_, 0);
        auto max_val = llaisys::Tensor::create({1}, meta_.dtype, device_, 0);
        llaisys::ops::argmax(max_idx, max_val, logits);

        // Argmax kernel writes size_t; on 64-bit this fits in I64 storage.
        const size_t idx = *reinterpret_cast<const size_t *>(max_idx->data());
        cur_len_ = ntoken;
        return static_cast<int64_t>(idx);
    } else {
        // decode
        CHECK_ARGUMENT(ntoken == cur_len_ + 1, "Qwen2Model: ntoken error");
        ensure_kv_cache_allocated_(k_cache_, v_cache_, meta_, device_, 0);

        // index: [1] i64
        int64_t last_token_id = token_ids[ntoken - 1];
        CHECK_ARGUMENT(last_token_id >= 0 && static_cast<size_t>(last_token_id) < meta_.voc, "Qwen2ModelInfer(decode): token id out of range");
        auto index = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device_, 0);
        index->load(&last_token_id);

        // x: [1, H]
        auto x = llaisys::Tensor::create({1, meta_.hs}, meta_.dtype, device_, 0);
        llaisys::ops::embedding(x, index, unwrap_(weights_.in_embed, "Qwen2Model: weights.in_embed is null"));

        // pos_ids: [1] i64
        auto pos_ids = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device_, 0);
        int64_t pos = ntoken - 1;
        CHECK_ARGUMENT(pos >= 0 && static_cast<size_t>(pos) < meta_.maxseq, "Qwen2ModelInfer(decode): pos exceeds maxseq");
        pos_ids->load(&pos);
        for (size_t layer = 0; layer < meta_.nlayer; ++layer) {
            CHECK_ARGUMENT(k_cache_[layer] != nullptr && v_cache_[layer] != nullptr, "Qwen2ModelInfer(decode): kv cache is null");

            // attn_norm:[1,H]
            auto x_attn_in = llaisys::Tensor::create({1, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::rms_norm(x_attn_in, x, unwrap_(weights_.attn_norm_w[layer], "Qwen2Model: weights.attn_norm_w is null"), meta_.epsilon);

            // q2d:[1,H]
            auto q2d = llaisys::Tensor::create({1, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::linear(
                q2d,
                x_attn_in,
                unwrap_(weights_.attn_q_w[layer], "Qwen2Model: weights.attn_q_w is null"),
                unwrap_(weights_.attn_q_b[layer], "Qwen2Model: weights.attn_q_b is null"));

            // k2d:[1,nkvh*hd]
            auto k2d = llaisys::Tensor::create({1, meta_.nkvh * meta_.dh}, meta_.dtype, device_, 0);
            llaisys::ops::linear(
                k2d,
                x_attn_in,
                unwrap_(weights_.attn_k_w[layer], "Qwen2Model: weights.attn_k_w is null"),
                unwrap_(weights_.attn_k_b[layer], "Qwen2Model: weights.attn_k_b is null"));

            // v2d:[1,nkvh*hd]
            auto v2d = llaisys::Tensor::create({1, meta_.nkvh * meta_.dh}, meta_.dtype, device_, 0);
            llaisys::ops::linear(
                v2d,
                x_attn_in,
                unwrap_(weights_.attn_v_w[layer], "Qwen2Model: weights.attn_v_w is null"),
                unwrap_(weights_.attn_v_b[layer], "Qwen2Model: weights.attn_v_b is null"));

            // q:[1,nh,nd]
            auto q = q2d->view({1, meta_.nh, meta_.dh});

            // k:[1,nkvh,nd]
            auto k = k2d->view({1, meta_.nkvh, meta_.dh});

            // v:[1,nkvh,nd]
            auto v = v2d->view({1, meta_.nkvh, meta_.dh});
            kv_cache_write_row_(v_cache_[layer], v, pos, meta_.nkvh, meta_.dh);

            // q_rope [1, nh, hd]
            auto q_rope = llaisys::Tensor::create({1, meta_.nh, meta_.dh}, meta_.dtype, device_, 0);
            llaisys::ops::rope(q_rope, q, pos_ids, meta_.theta);

            // k_rope [1, nkvh, hd]
            auto k_rope = llaisys::Tensor::create({1, meta_.nkvh, meta_.dh}, meta_.dtype, device_, 0);
            llaisys::ops::rope(k_rope, k, pos_ids, meta_.theta);
            kv_cache_write_row_(k_cache_[layer], k_rope, pos, meta_.nkvh, meta_.dh);

            // attn_val [1,nh,hd]
            auto attn_val = llaisys::Tensor::create({1, meta_.nh, meta_.dh}, meta_.dtype, device_, 0);

            size_t kvlen = pos + 1;
            CHECK_ARGUMENT(kvlen <= meta_.maxseq, "Qwen2ModelInfer(decode): kvlen exceeds maxseq");
            // k_prefix [kvlen,nkvh,hd]
            auto k_prefix = k_cache_[layer]->slice(0, 0, kvlen);

            // v_prefix [kvlen,nkvh,hd]
            auto v_prefix = v_cache_[layer]->slice(0, 0, kvlen);
            llaisys::ops::self_attention(attn_val, q_rope, k_prefix, v_prefix, float(1.0f / std::sqrt(meta_.dh)));

            // attn2d [1,H]
            auto attn2d = attn_val->view({1, meta_.hs});

            // attn_out [1,H]
            auto attn_out = llaisys::Tensor::create({1, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::linear(attn_out, attn2d, unwrap_(weights_.attn_o_w[layer], "Qwen2Model: weights.attn_o_ws is null"), nullptr);

            // x_after_attn [1,H]
            auto x_after_attn = llaisys::Tensor::create({1, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::add(x_after_attn, x, attn_out);

            // x_mlp_in [1,H]
            auto x_mlp_in = llaisys::Tensor::create({1, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::rms_norm(x_mlp_in, x_after_attn, unwrap_(weights_.mlp_norm_w[layer], "Qwen2Model: weights.mlp_norm_w is null"), meta_.epsilon);

            // gate [1,di]
            auto gate = llaisys::Tensor::create({1, meta_.di}, meta_.dtype, device_, 0);
            llaisys::ops::linear(gate, x_mlp_in, unwrap_(weights_.mlp_gate_w[layer], "Qwen2Model: weights_.mlp_gate_w is null"), nullptr);

            // up [1,di]
            auto up = llaisys::Tensor::create({1, meta_.di}, meta_.dtype, device_, 0);
            llaisys::ops::linear(up, x_mlp_in, unwrap_(weights_.mlp_up_w[layer], "Qwen2Model: weights_.mlp_up_w is null"), nullptr);

            // act [1,di]
            auto act = llaisys::Tensor::create({1, meta_.di}, meta_.dtype, device_, 0);
            llaisys::ops::swiglu(act, gate, up);

            // mlp_out [1,H]
            auto mlp_out = llaisys::Tensor::create({1, meta_.hs}, meta_.dtype, device_, 0);
            llaisys::ops::linear(mlp_out, act, unwrap_(weights_.mlp_down_w[layer], "Qwen2Model: weights_.mlp_down_w is null"), nullptr);

            // x_after_mlp [1,H]
            llaisys::ops::add(x, x_after_attn, mlp_out);
        }

        // final norm: [1, H]
        auto last_x = x->slice(0, 0, 1);
        auto x_norm = llaisys::Tensor::create({1, meta_.hs}, meta_.dtype, device_, 0);
        llaisys::ops::rms_norm(x_norm, last_x, unwrap_(weights_.out_norm_w, "Qwen2Model: weights.out_norm_w is null"), meta_.epsilon);

        // logits2d: [1, vocab]
        auto logits2d = llaisys::Tensor::create({1, meta_.voc}, meta_.dtype, device_, 0);
        llaisys::ops::linear(
            logits2d,
            x_norm,
            unwrap_(weights_.out_embed, "Qwen2Model: weights.out_embed is null"),
            nullptr);

        // argmax over vocab (treat logits as 1D [vocab])
        auto logits = logits2d->view({meta_.voc});
        auto max_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device_, 0);
        auto max_val = llaisys::Tensor::create({1}, meta_.dtype, device_, 0);
        llaisys::ops::argmax(max_idx, max_val, logits);

        // Argmax kernel writes size_t; on 64-bit this fits in I64 storage.
        const size_t idx = *reinterpret_cast<const size_t *>(max_idx->data());
        cur_len_ = ntoken;
        return static_cast<int64_t>(idx);
    }
}

const char *Qwen2Model::missingWeightsHint_() const {
    if (weights_.in_embed == nullptr) {
        return "Qwen2Model: missing weights.in_embed (model.embed_tokens.weight)";
    }
    if (weights_.out_embed == nullptr) {
        return "Qwen2Model: missing weights.out_embed (lm_head.weight)";
    }
    if (weights_.out_norm_w == nullptr) {
        return "Qwen2Model: missing weights.out_norm_w (model.norm.weight)";
    }

    // Spot-check layer 0 to catch mapping bugs early.
    if (!weights_.attn_norm_w || weights_.attn_norm_w[0] == nullptr) {
        return "Qwen2Model: missing weights.attn_norm_w[0]";
    }
    if (!weights_.attn_q_w || weights_.attn_q_w[0] == nullptr) {
        return "Qwen2Model: missing weights.attn_q_w[0]";
    }
    if (!weights_.attn_k_w || weights_.attn_k_w[0] == nullptr) {
        return "Qwen2Model: missing weights.attn_k_w[0]";
    }
    if (!weights_.attn_v_w || weights_.attn_v_w[0] == nullptr) {
        return "Qwen2Model: missing weights.attn_v_w[0]";
    }
    if (!weights_.attn_o_w || weights_.attn_o_w[0] == nullptr) {
        return "Qwen2Model: missing weights.attn_o_w[0]";
    }

    if (!weights_.mlp_norm_w || weights_.mlp_norm_w[0] == nullptr) {
        return "Qwen2Model: missing weights.mlp_norm_w[0]";
    }
    if (!weights_.mlp_gate_w || weights_.mlp_gate_w[0] == nullptr) {
        return "Qwen2Model: missing weights.mlp_gate_w[0]";
    }
    if (!weights_.mlp_up_w || weights_.mlp_up_w[0] == nullptr) {
        return "Qwen2Model: missing weights.mlp_up_w[0]";
    }
    if (!weights_.mlp_down_w || weights_.mlp_down_w[0] == nullptr) {
        return "Qwen2Model: missing weights.mlp_down_w[0]";
    }

    return nullptr;
}

} // namespace llaisys::models
