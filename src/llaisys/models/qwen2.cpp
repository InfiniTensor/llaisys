// src/llaisys/qwen2.cpp

#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include "../../utils.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/rearrange/op.hpp"


#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <cstring> // std::memcpy
#include <cmath>


using llaisys::Tensor;
using llaisys::tensor_t;

namespace {

// Per-layer KV cache
struct LayerCache {
    tensor_t K; // [maxseq, nkvh, dh]
    tensor_t V; // [maxseq, nkvh, dh]  (we use dv == dh)
};

// Model holder
struct Model {
    LlaisysQwen2Meta meta{};
    std::unordered_map<std::string, tensor_t> W; // name -> tensor

    // KV cache
    std::vector<LayerCache> cache;
    size_t cur_len = 0; // how many tokens are cached

    // Reusable small tensors to reduce allocations
    tensor_t idx1;    // [1] i64
    tensor_t pos1;    // [1] i64
    tensor_t x1;      // [1, hs]
    tensor_t h1;      // [1, hs]
    tensor_t h2;      // [1, hs]
    tensor_t q2;      // [1, nh*dh]
    tensor_t k2;      // [1, nkvh*dh]
    tensor_t v2;      // [1, nkvh*dh]  (dv == dh)
    tensor_t Q;       // [1, nh,   dh]
    tensor_t K;       // [1, nkvh, dh]
    tensor_t V;       // [1, nkvh, dh]
    tensor_t Qr;      // [1, nh,   dh]
    tensor_t Kr;      // [1, nkvh, dh]
    tensor_t Yatt;    // [1, nh,   dh]
    tensor_t Yproj;   // [1, hs]
    tensor_t G;       // [1, di]
    tensor_t U;       // [1, di]
    tensor_t GU;      // [1, di]
    tensor_t Dn;      // [1, hs]
    tensor_t Xn1;     // [1, hs]
    tensor_t Logits1; // [1, voc]

    tensor_t T(const std::vector<size_t> &shape, llaisysDataType_t dt) {
        return Tensor::create(shape, dt, LLAISYS_DEVICE_CPU, 0);
    }

    void ensure_buffers() {
        const auto &m = meta;
        if (!idx1) {
            idx1 = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        }
        if (!pos1) {
            pos1 = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        }
        if (!x1) {
            x1 = T({1, m.hs}, m.dtype);
        }
        if (!h1) {
            h1 = T({1, m.hs}, m.dtype);
        }
        if (!h2) {
            h2 = T({1, m.hs}, m.dtype);
        }
        if (!q2) {
            q2 = T({1, m.nh * m.dh}, m.dtype);
        }
        if (!k2) {
            k2 = T({1, m.nkvh * m.dh}, m.dtype);
        }
        if (!v2) {
            v2 = T({1, m.nkvh * m.dh}, m.dtype); // dv == dh
        }
        if (!Q) {
            Q = T({1, m.nh, m.dh}, m.dtype);
        }
        if (!K) {
            K = T({1, m.nkvh, m.dh}, m.dtype);
        }
        if (!V) {
            V = T({1, m.nkvh, m.dh}, m.dtype);
        }
        if (!Qr) {
            Qr = T({1, m.nh, m.dh}, m.dtype);
        }
        if (!Kr) {
            Kr = T({1, m.nkvh, m.dh}, m.dtype);
        }
        if (!Yatt) {
            Yatt = T({1, m.nh, m.dh}, m.dtype);
        }
        if (!Yproj) {
            Yproj = T({1, m.hs}, m.dtype);
        }
        if (!G) {
            G = T({1, m.di}, m.dtype);
        }
        if (!U) {
            U = T({1, m.di}, m.dtype);
        }
        if (!GU) {
            GU = T({1, m.di}, m.dtype);
        }
        if (!Dn) {
            Dn = T({1, m.hs}, m.dtype);
        }
        if (!Xn1) {
            Xn1 = T({1, m.hs}, m.dtype);
        }
        if (!Logits1) {
            Logits1 = T({1, m.voc}, m.dtype);
        }
    }

    void ensure_cache() {
        if (!cache.empty()) {
            return;
        }
        const auto &m = meta;
        cache.resize(m.nlayer);
        for (size_t l = 0; l < m.nlayer; ++l) {
            cache[l].K = T({m.maxseq, m.nkvh, m.dh}, m.dtype);
            cache[l].V = T({m.maxseq, m.nkvh, m.dh}, m.dtype);
        }
        cur_len = 0;
    }
};

inline tensor_t getW(Model *M, const std::string &name) {
    auto it = M->W.find(name);
    ASSERT(it != M->W.end(), ("missing weight: " + name).c_str());
    return it->second;
}

inline tensor_t as_qkv_3d(const tensor_t &x2d, size_t H, size_t D) {
    // x2d shape: [1, H*D]
    return x2d->view({x2d->shape()[0], H, D});
}

// Run one token through the whole stack, append K/V into cache, and return next id.
static int64_t step_decode_one(Model &M, int64_t token_id, size_t pos_idx) {
    const auto &meta = M.meta;
    M.ensure_buffers();
    M.ensure_cache();

    // Embedding for [token_id] -> x1: [1, hs]
    M.idx1->load(&token_id);
    llaisys::ops::embedding(M.x1, M.idx1, getW(&M, "model.embed_tokens.weight"));

    // Layers
    for (size_t l = 0; l < meta.nlayer; ++l) {
        // Pre-attention RMSNorm
        auto ln = getW(&M, "model.layers." + std::to_string(l) + ".input_layernorm.weight");
        llaisys::ops::rms_norm(M.h1, M.x1, ln, meta.epsilon);

        // Q, K, V projections
        llaisys::ops::linear(M.q2, M.h1, getW(&M, "model.layers." + std::to_string(l) + ".self_attn.q_proj.weight"), nullptr);
        llaisys::ops::linear(M.k2, M.h1, getW(&M, "model.layers." + std::to_string(l) + ".self_attn.k_proj.weight"), nullptr);
        llaisys::ops::linear(M.v2, M.h1, getW(&M, "model.layers." + std::to_string(l) + ".self_attn.v_proj.weight"), nullptr);

        M.Q = as_qkv_3d(M.q2, meta.nh, meta.dh);
        M.K = as_qkv_3d(M.k2, meta.nkvh, meta.dh);
        M.V = as_qkv_3d(M.v2, meta.nkvh, meta.dh);

        // RoPE with position pos_idx
        int64_t p = static_cast<int64_t>(pos_idx);
        M.pos1->load(&p);
        llaisys::ops::rope(M.Qr, M.Q, M.pos1, meta.theta);
        llaisys::ops::rope(M.Kr, M.K, M.pos1, meta.theta);

        // Write this token's K/V into cache row pos_idx
        auto K_row = M.cache[l].K->slice(0, pos_idx, pos_idx + 1); // [1, nkvh, dh]
        auto V_row = M.cache[l].V->slice(0, pos_idx, pos_idx + 1); // [1, nkvh, dh]
        /*llaisys::ops::rearrange(K_row, M.Kr);
        llaisys::ops::rearrange(V_row, M.V);*/
         if (K_row->isContiguous() && M.Kr->isContiguous()) {
             const size_t bytes = K_row->numel() * K_row->elementSize();
             std::memcpy(K_row->data(), M.Kr->data(), bytes);
         } else {
             llaisys::ops::rearrange(K_row, M.Kr);
         }
         if (V_row->isContiguous() && M.V->isContiguous()) {
             const size_t bytes = V_row->numel() * V_row->elementSize();
             std::memcpy(V_row->data(), M.V->data(), bytes);
         } else {
             llaisys::ops::rearrange(V_row, M.V);
         }

        // Attention for query at current position against K/V up to pos_idx
        auto K_used = M.cache[l].K->slice(0, 0, pos_idx + 1); // [pos_idx+1, nkvh, dh]
        auto V_used = M.cache[l].V->slice(0, 0, pos_idx + 1); // [pos_idx+1, nkvh, dh]
        const float scale = 1.0f / std::sqrt((float)meta.dh);
        llaisys::ops::self_attention(M.Yatt, M.Qr, K_used, V_used, scale); // [1, nh, dh]

        // Output projection and residual
        llaisys::ops::linear(M.Yproj, M.Yatt->view({1, meta.nh * meta.dh}),
                             getW(&M, "model.layers." + std::to_string(l) + ".self_attn.o_proj.weight"),
                             nullptr);
        llaisys::ops::add(M.x1, M.x1, M.Yproj);

        // MLP path: norm -> gate/up -> swiglu -> down -> residual
        auto ln2 = getW(&M, "model.layers." + std::to_string(l) + ".post_attention_layernorm.weight");
        llaisys::ops::rms_norm(M.h2, M.x1, ln2, meta.epsilon);

        llaisys::ops::linear(M.G, M.h2, getW(&M, "model.layers." + std::to_string(l) + ".mlp.gate_proj.weight"), nullptr);
        llaisys::ops::linear(M.U, M.h2, getW(&M, "model.layers." + std::to_string(l) + ".mlp.up_proj.weight"), nullptr);
        llaisys::ops::swiglu(M.GU, M.G, M.U);

        llaisys::ops::linear(M.Dn, M.GU, getW(&M, "model.layers." + std::to_string(l) + ".mlp.down_proj.weight"), nullptr);
        llaisys::ops::add(M.x1, M.x1, M.Dn);
    }

    // Final RMSNorm and lm_head for a single row
    llaisys::ops::rms_norm(M.Xn1, M.x1, getW(&M, "model.norm.weight"), meta.epsilon);
    llaisys::ops::linear(M.Logits1, M.Xn1, getW(&M, "lm_head.weight"), nullptr); // [1, voc]

    auto last1d = M.Logits1->view({meta.voc});
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    auto max_val = Tensor::create({1}, meta.dtype, LLAISYS_DEVICE_CPU, 0);
    llaisys::ops::argmax(max_idx, max_val, last1d);

    int64_t next_id = 0;
    std::memcpy(&next_id, max_idx->data(), sizeof(int64_t));
    return next_id;
}

} // namespace

extern "C" {

struct LlaisysQwen2Model {
    Model impl;
};

__export LlaisysQwen2Model *
llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                        llaisysDeviceType_t device,
                        int *device_ids, int ndevice) {
    (void)device;
    (void)device_ids;
    (void)ndevice;
    auto *h = new LlaisysQwen2Model();
    h->impl.meta = *meta;
    return h;
}

__export void
llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    delete model;
}

__export int
llaisysQwen2ModelLoadNamedWeight(LlaisysQwen2Model *model,
                                 const char *name,
                                 const void *data,
                                 const size_t *shape,
                                 size_t ndim,
                                 llaisysDataType_t dtype) {
    auto &M = model->impl;
    std::string key(name);
    std::vector<size_t> shp(shape, shape + ndim);

    auto t = Tensor::create(shp, dtype, LLAISYS_DEVICE_CPU, 0);
    t->load(data);
    M.W[key] = t;
    return 0;
}

__export int64_t
llaisysQwen2ModelInfer(LlaisysQwen2Model *model,
                       const int64_t *token_ids,
                       size_t ntok) {
    auto &M = model->impl;
    const auto &meta = M.meta;

    M.ensure_buffers();
    M.ensure_cache();
    int64_t next_id = -1;

    // Prefill missing prefix into KV cache
    for (size_t t = M.cur_len; t < ntok; ++t) {
        int64_t tid = token_ids[t];
        next_id = step_decode_one(M, tid, t);
        M.cur_len = t + 1;
        if (M.cur_len >= meta.maxseq) {
            break;
        }
    }
    std::cerr << "[qwen2] layer "  << " done\n";

    // Compute next id using the last token and existing KV
    if (next_id == -1) {
        const size_t last = ntok - 1;
        next_id = step_decode_one(M, token_ids[last], last);
    }
    return next_id;
}

} // extern "C"
