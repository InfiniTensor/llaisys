#include "llaisys/models/qwen2.h"

#include "../llaisys_tensor.hpp"
#include "../../core/context/context.hpp"
#include "../../ops/ops.hpp"
#include "../../tensor/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace {
// trigger actions

// Helper: convert llaisysTensor_t (C handle) -> internal tensor_t
inline llaisys::tensor_t as_tensor(llaisysTensor_t t) {
    if (!t) return nullptr;
    return t->tensor;
}

inline llaisys::tensor_t make_tensor(const std::vector<size_t> &shape,
                                     llaisysDataType_t dtype,
                                     llaisysDeviceType_t device,
                                     int device_id) {
    return llaisys::Tensor::create(shape, dtype, device, device_id);
}

inline llaisys::tensor_t make_i64_tensor_1d(size_t n,
                                            llaisysDeviceType_t device,
                                            int device_id) {
    return make_tensor({n}, LLAISYS_DTYPE_I64, device, device_id);
}

// Fill an int64 tensor (contiguous 1D) from host vector
inline void load_i64_1d(llaisys::tensor_t t, const std::vector<int64_t> &vals) {
    if (!t) throw std::runtime_error("load_i64_1d: tensor is null");
    if (t->dtype() != LLAISYS_DTYPE_I64) throw std::runtime_error("load_i64_1d: dtype not I64");
    if (t->ndim() != 1) throw std::runtime_error("load_i64_1d: ndim not 1");
    if (t->shape()[0] != vals.size()) throw std::runtime_error("load_i64_1d: shape mismatch");
    t->load(vals.data());
}

// Slice cache along dim0 to total_len: [0:total_len)
inline llaisys::tensor_t slice0(llaisys::tensor_t t, size_t total_len) {
    return t->slice(0, 0, total_len);
}

// View helper (assumes compatible contiguous)
inline llaisys::tensor_t view(llaisys::tensor_t t, const std::vector<size_t> &shape) {
    return t->view(shape);
}

inline void copy_bytes(void *dst,
                       const void *src,
                       size_t bytes,
                       llaisysMemcpyKind_t kind,
                       llaisysDeviceType_t device,
                       int device_id) {
    llaisys::core::context().setDevice(device, device_id);
    llaisys::core::context().runtime().api()->memcpy_sync(dst, src, bytes, kind);
}

inline int64_t read_i64_scalar(llaisys::tensor_t t,
                               llaisysDeviceType_t device,
                               int device_id) {
    int64_t host_val = -1;
    if (device == LLAISYS_DEVICE_CPU) {
        std::memcpy(&host_val, t->data(), sizeof(host_val));
    } else {
        copy_bytes(&host_val, t->data(), sizeof(host_val), LLAISYS_MEMCPY_D2H, device, device_id);
    }
    return host_val;
}

} // namespace

__C {

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    llaisysDeviceType_t device{LLAISYS_DEVICE_CPU};
    int device_id{0};

    // weights (C handles + arrays)
    LlaisysQwen2Weights weights{};

    // persistent KV cache (internal tensors)
    std::vector<llaisys::tensor_t> k_cache; // [maxseq, nkvh, dh]
    std::vector<llaisys::tensor_t> v_cache; // [maxseq, nkvh, dh]
    size_t past_len{0};                     // how many tokens already cached

    // For convenience: keep allocated arrays for weights struct to free on destroy
    std::vector<llaisysTensor_t> attn_norm_w_arr;
    std::vector<llaisysTensor_t> attn_q_w_arr;
    std::vector<llaisysTensor_t> attn_q_b_arr;
    std::vector<llaisysTensor_t> attn_k_w_arr;
    std::vector<llaisysTensor_t> attn_k_b_arr;
    std::vector<llaisysTensor_t> attn_v_w_arr;
    std::vector<llaisysTensor_t> attn_v_b_arr;
    std::vector<llaisysTensor_t> attn_o_w_arr;

    std::vector<llaisysTensor_t> mlp_norm_w_arr;
    std::vector<llaisysTensor_t> mlp_gate_w_arr;
    std::vector<llaisysTensor_t> mlp_up_w_arr;
    std::vector<llaisysTensor_t> mlp_down_w_arr;
};

static void init_weight_arrays(LlaisysQwen2Model *m) {
    size_t n = m->meta.nlayer;

    m->attn_norm_w_arr.assign(n, nullptr);
    m->attn_q_w_arr.assign(n, nullptr);
    m->attn_q_b_arr.assign(n, nullptr);
    m->attn_k_w_arr.assign(n, nullptr);
    m->attn_k_b_arr.assign(n, nullptr);
    m->attn_v_w_arr.assign(n, nullptr);
    m->attn_v_b_arr.assign(n, nullptr);
    m->attn_o_w_arr.assign(n, nullptr);

    m->mlp_norm_w_arr.assign(n, nullptr);
    m->mlp_gate_w_arr.assign(n, nullptr);
    m->mlp_up_w_arr.assign(n, nullptr);
    m->mlp_down_w_arr.assign(n, nullptr);

    // expose pointers in weights struct
    m->weights.attn_norm_w = m->attn_norm_w_arr.data();
    m->weights.attn_q_w    = m->attn_q_w_arr.data();
    m->weights.attn_q_b    = m->attn_q_b_arr.data();
    m->weights.attn_k_w    = m->attn_k_w_arr.data();
    m->weights.attn_k_b    = m->attn_k_b_arr.data();
    m->weights.attn_v_w    = m->attn_v_w_arr.data();
    m->weights.attn_v_b    = m->attn_v_b_arr.data();
    m->weights.attn_o_w    = m->attn_o_w_arr.data();

    m->weights.mlp_norm_w  = m->mlp_norm_w_arr.data();
    m->weights.mlp_gate_w  = m->mlp_gate_w_arr.data();
    m->weights.mlp_up_w    = m->mlp_up_w_arr.data();
    m->weights.mlp_down_w  = m->mlp_down_w_arr.data();
}

static void init_kv_cache(LlaisysQwen2Model *m) {
    size_t nlayer = m->meta.nlayer;
    size_t maxseq = m->meta.maxseq;
    size_t nkvh   = m->meta.nkvh;
    size_t dh     = m->meta.dh;

    m->k_cache.resize(nlayer);
    m->v_cache.resize(nlayer);

    for (size_t l = 0; l < nlayer; ++l) {
        m->k_cache[l] = make_tensor({maxseq, nkvh, dh}, m->meta.dtype, m->device, m->device_id);
        m->v_cache[l] = make_tensor({maxseq, nkvh, dh}, m->meta.dtype, m->device, m->device_id);
    }

    m->past_len = 0;
}

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int * /*device_ids*/,
    int /*ndevice*/
) {
    if (!meta) return nullptr;

    auto *m = new LlaisysQwen2Model();
    m->meta = *meta;
    m->device = device;
    m->device_id = 0;

    init_weight_arrays(m);
    init_kv_cache(m);

    // weights.in_embed/out_embed/out_norm_w will be filled by Python loader
    m->weights.in_embed = nullptr;
    m->weights.out_embed = nullptr;
    m->weights.out_norm_w = nullptr;

    return m;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) return;
    delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model) return nullptr;
    return &model->weights;
}

__export void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model) {
    if (!model) return;
    model->past_len = 0;
}

} // __C

// One forward pass on a chunk of "new tokens" of length seqlen.
// It consumes tokens, writes KV into cache, and returns logits for last position.
// 这是 C++ 内部辅助函数，不需要导出成 C API；放到 extern "C" 之外可避免 MSVC 把返回 shared_ptr 当成 C linkage 警告。
static llaisys::tensor_t qwen2_forward_last_logits(LlaisysQwen2Model *m,
                                                   const int64_t *token_ids,
                                                   size_t seqlen) {
    if (!m) return nullptr;
    if (!token_ids || seqlen == 0) return nullptr;

    const auto &meta = m->meta;
    size_t hs   = meta.hs;
    size_t nh   = meta.nh;
    size_t nkvh = meta.nkvh;
    size_t dh   = meta.dh;
    size_t di   = meta.di;
    size_t voc  = meta.voc;

    if (m->past_len + seqlen > meta.maxseq) return nullptr;

    // ---- Build index tensor [seqlen] int64 ----
    auto idx = make_i64_tensor_1d(seqlen, m->device, m->device_id);
    idx->load(token_ids);

    // ---- Embedding: x = embed(tokens) -> [seqlen, hs] ----
    if (!m->weights.in_embed) return nullptr;
    auto x = make_tensor({seqlen, hs}, meta.dtype, m->device, m->device_id);
    llaisys::ops::embedding(x, idx, as_tensor(m->weights.in_embed));

    // position ids: [seqlen], values [past_len ... past_len+seqlen-1]
    std::vector<int64_t> pos_host(seqlen);
    for (size_t i = 0; i < seqlen; ++i) pos_host[i] = (int64_t)(m->past_len + i);
    auto pos_ids = make_i64_tensor_1d(seqlen, m->device, m->device_id);
    load_i64_1d(pos_ids, pos_host);

    // ---- Transformer blocks ----
    for (size_t l = 0; l < meta.nlayer; ++l) {
        if (!m->weights.attn_norm_w[l]) return nullptr;
        if (!m->weights.attn_q_w[l]) return nullptr;
        if (!m->weights.attn_k_w[l]) return nullptr;
        if (!m->weights.attn_v_w[l]) return nullptr;
        if (!m->weights.attn_o_w[l]) return nullptr;
        if (!m->weights.mlp_norm_w[l]) return nullptr;
        if (!m->weights.mlp_gate_w[l]) return nullptr;
        if (!m->weights.mlp_up_w[l]) return nullptr;
        if (!m->weights.mlp_down_w[l]) return nullptr;

        // 1) attn rmsnorm
        auto x_norm = make_tensor({seqlen, hs}, meta.dtype, m->device, m->device_id);
        llaisys::ops::rms_norm(x_norm, x, as_tensor(m->weights.attn_norm_w[l]), meta.epsilon);

        // 2) q/k/v projections
        auto q_lin = make_tensor({seqlen, hs}, meta.dtype, m->device, m->device_id);
        llaisys::ops::linear(q_lin, x_norm, as_tensor(m->weights.attn_q_w[l]), as_tensor(m->weights.attn_q_b[l]));

        auto k_lin = make_tensor({seqlen, nkvh * dh}, meta.dtype, m->device, m->device_id);
        auto v_lin = make_tensor({seqlen, nkvh * dh}, meta.dtype, m->device, m->device_id);
        llaisys::ops::linear(k_lin, x_norm, as_tensor(m->weights.attn_k_w[l]), as_tensor(m->weights.attn_k_b[l]));
        llaisys::ops::linear(v_lin, x_norm, as_tensor(m->weights.attn_v_w[l]), as_tensor(m->weights.attn_v_b[l]));

        // 3) reshape to [seqlen, head, dh]
        auto q = view(q_lin, {seqlen, nh, dh});
        auto k = view(k_lin, {seqlen, nkvh, dh});
        auto v = view(v_lin, {seqlen, nkvh, dh});

        // 4) rope on q,k
        auto q_rope = make_tensor({seqlen, nh, dh}, meta.dtype, m->device, m->device_id);
        auto k_rope = make_tensor({seqlen, nkvh, dh}, meta.dtype, m->device, m->device_id);
        llaisys::ops::rope(q_rope, q, pos_ids, meta.theta);
        llaisys::ops::rope(k_rope, k, pos_ids, meta.theta);

        // 5) write k/v into cache at [past_len : past_len + seqlen)
        {
            auto k_dst = m->k_cache[l]->slice(0, m->past_len, m->past_len + seqlen);
            auto v_dst = m->v_cache[l]->slice(0, m->past_len, m->past_len + seqlen);

            copy_bytes(k_dst->data(), k_rope->data(), k_rope->numel() * k_rope->elementSize(), LLAISYS_MEMCPY_D2D, m->device, m->device_id);
            copy_bytes(v_dst->data(), v->data(), v->numel() * v->elementSize(), LLAISYS_MEMCPY_D2D, m->device, m->device_id);
        }

        size_t total_len = m->past_len + seqlen;

        // 6) self attention: attn_val [seqlen, nh, dh]
        auto k_total = slice0(m->k_cache[l], total_len);
        auto v_total = slice0(m->v_cache[l], total_len);

        auto attn_val = make_tensor({seqlen, nh, dh}, meta.dtype, m->device, m->device_id);
        float scale = 1.0f / std::sqrt((float)dh);
        llaisys::ops::self_attention(attn_val, q_rope, k_total, v_total, scale);

        // 7) merge heads -> [seqlen, hs]
        auto attn_merge = view(attn_val, {seqlen, hs});

        // 8) output proj
        auto attn_out = make_tensor({seqlen, hs}, meta.dtype, m->device, m->device_id);
        llaisys::ops::linear(attn_out, attn_merge, as_tensor(m->weights.attn_o_w[l]), nullptr);

        // 9) residual add
        auto x_attn = make_tensor({seqlen, hs}, meta.dtype, m->device, m->device_id);
        llaisys::ops::add(x_attn, x, attn_out);
        x = x_attn;

        // 10) mlp rmsnorm
        auto x_mlp_norm = make_tensor({seqlen, hs}, meta.dtype, m->device, m->device_id);
        llaisys::ops::rms_norm(x_mlp_norm, x, as_tensor(m->weights.mlp_norm_w[l]), meta.epsilon);

        // 11) gate/up
        auto gate = make_tensor({seqlen, di}, meta.dtype, m->device, m->device_id);
        auto up   = make_tensor({seqlen, di}, meta.dtype, m->device, m->device_id);
        llaisys::ops::linear(gate, x_mlp_norm, as_tensor(m->weights.mlp_gate_w[l]), nullptr);
        llaisys::ops::linear(up,   x_mlp_norm, as_tensor(m->weights.mlp_up_w[l]),   nullptr);

        // 12) swiglu
        auto act = make_tensor({seqlen, di}, meta.dtype, m->device, m->device_id);
        llaisys::ops::swiglu(act, gate, up);

        // 13) down proj
        auto down = make_tensor({seqlen, hs}, meta.dtype, m->device, m->device_id);
        llaisys::ops::linear(down, act, as_tensor(m->weights.mlp_down_w[l]), nullptr);

        // 14) residual add
        auto x_mlp = make_tensor({seqlen, hs}, meta.dtype, m->device, m->device_id);
        llaisys::ops::add(x_mlp, x, down);
        x = x_mlp;
    }

    // update cache length AFTER processing this chunk
    m->past_len += seqlen;

    // final norm
    if (!m->weights.out_norm_w) return nullptr;
    auto x_final = make_tensor({seqlen, hs}, meta.dtype, m->device, m->device_id);
    llaisys::ops::rms_norm(x_final, x, as_tensor(m->weights.out_norm_w), meta.epsilon);

    // logits
    if (!m->weights.out_embed) return nullptr;
    auto logits = make_tensor({seqlen, voc}, meta.dtype, m->device, m->device_id);
    llaisys::ops::linear(logits, x_final, as_tensor(m->weights.out_embed), nullptr);

    return logits->slice(0, seqlen - 1, seqlen)->view({voc});
}

__C {

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model,
                                       int64_t *token_ids,
                                       size_t ntoken) {
    auto logits = qwen2_forward_last_logits(model, token_ids, ntoken);
    if (!logits) return -1;
    auto max_idx = make_tensor({1}, LLAISYS_DTYPE_I64, model->device, model->device_id);
    auto max_val = make_tensor({1}, model->meta.dtype, model->device, model->device_id);
    llaisys::ops::argmax(max_idx, max_val, logits);
    return read_i64_scalar(max_idx, model->device, model->device_id);
}

__export int64_t llaisysQwen2ModelInferSample(struct LlaisysQwen2Model *model,
                                              int64_t *token_ids,
                                              size_t ntoken,
                                              float temperature,
                                              int top_k,
                                              float top_p,
                                              uint64_t seed) {
    auto logits = qwen2_forward_last_logits(model, token_ids, ntoken);
    if (!logits) return -1;
    return llaisys::ops::sample(logits, temperature, top_k, top_p, seed);
}


} // __C
