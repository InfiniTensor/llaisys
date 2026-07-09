#include "llaisys/models/qwen2.h"
#include "../ops/ops.hpp"
#include "../utils.hpp"
#include "llaisys_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

namespace llaisys {

// ─── BF16/F16 → F32 bit-cast helpers（logit 类型转换，仅 CPU）───────────────
static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign << 31;
        } else {
            exp = 1u;
            while (!(mant & 0x400u)) {
                mant <<= 1u;
                exp--;
            }
            mant &= 0x3ffu;
            bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13u);
        }
    } else if (exp == 31u) {
        bits = (sign << 31) | (0xffu << 23) | (mant << 13u);
    } else {
        bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13u);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

// ─── 每用户 Session：独立 KV-Cache + 采样状态 ────────────────────────────────
// 权重只读共享，多个 Session 可并发操作同一 Qwen2Model 而互不干扰。
struct Qwen2Session {
    LlaisysQwen2Meta meta; // 模型 metadata 副本（用于分配 tensor）
    llaisysDeviceType_t device;
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    size_t cache_pos = 0;
    std::mt19937 rng;

    Qwen2Session(const LlaisysQwen2Meta &m, llaisysDeviceType_t dev)
        : meta(m), device(dev), cache_pos(0), rng(std::random_device{}()) {
        k_cache.resize(meta.nlayer);
        v_cache.resize(meta.nlayer);
        for (size_t i = 0; i < meta.nlayer; i++) {
            k_cache[i] = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device);
            v_cache[i] = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device);
        }
    }

    void reset() {
        cache_pos = 0;
    }
    void set_pos(size_t pos) { cache_pos = pos; }
    size_t get_pos() const { return cache_pos; }
};

struct Qwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;

    // Weights
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;

    // 向后兼容的默认会话（单用户接口通过此 session 操作）
    std::unique_ptr<Qwen2Session> default_session;

    Qwen2Model(const LlaisysQwen2Meta &m, llaisysDeviceType_t dev)
        : meta(m), device(dev) {
        // Create embedding and output tensors
        in_embed = Tensor::create({meta.voc, meta.hs}, meta.dtype, device);
        out_embed = Tensor::create({meta.voc, meta.hs}, meta.dtype, device);
        out_norm_w = Tensor::create({meta.hs}, meta.dtype, device);

        // Initialize weight vectors
        attn_norm_w.resize(meta.nlayer);
        attn_q_w.resize(meta.nlayer);
        attn_q_b.resize(meta.nlayer);
        attn_k_w.resize(meta.nlayer);
        attn_k_b.resize(meta.nlayer);
        attn_v_w.resize(meta.nlayer);
        attn_v_b.resize(meta.nlayer);
        attn_o_w.resize(meta.nlayer);
        mlp_norm_w.resize(meta.nlayer);
        mlp_gate_w.resize(meta.nlayer);
        mlp_up_w.resize(meta.nlayer);
        mlp_down_w.resize(meta.nlayer);

        // Create per-layer weight tensors
        for (size_t i = 0; i < meta.nlayer; i++) {
            attn_norm_w[i] = Tensor::create({meta.hs}, meta.dtype, device);
            attn_q_w[i] = Tensor::create({meta.nh * meta.dh, meta.hs}, meta.dtype, device);
            attn_q_b[i] = Tensor::create({meta.nh * meta.dh}, meta.dtype, device);
            attn_k_w[i] = Tensor::create({meta.nkvh * meta.dh, meta.hs}, meta.dtype, device);
            attn_k_b[i] = Tensor::create({meta.nkvh * meta.dh}, meta.dtype, device);
            attn_v_w[i] = Tensor::create({meta.nkvh * meta.dh, meta.hs}, meta.dtype, device);
            attn_v_b[i] = Tensor::create({meta.nkvh * meta.dh}, meta.dtype, device);
            attn_o_w[i] = Tensor::create({meta.hs, meta.hs}, meta.dtype, device);
            mlp_norm_w[i] = Tensor::create({meta.hs}, meta.dtype, device);
            mlp_gate_w[i] = Tensor::create({meta.di, meta.hs}, meta.dtype, device);
            mlp_up_w[i] = Tensor::create({meta.di, meta.hs}, meta.dtype, device);
            mlp_down_w[i] = Tensor::create({meta.hs, meta.di}, meta.dtype, device);
        }

        // 创建默认会话（向后兼容单用户接口）
        default_session = std::make_unique<Qwen2Session>(m, dev);
    }

    // 运行完整前向传播，更新 sess.cache_pos，返回设备端 logits [voc]
    tensor_t run_forward(Qwen2Session &sess, int64_t *token_ids, size_t ntoken) {
        size_t seq_len = ntoken;

        // Create position ids
        auto pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device);
        std::vector<int64_t> pos_data(seq_len);
        for (size_t i = 0; i < seq_len; i++) {
            pos_data[i] = static_cast<int64_t>(sess.cache_pos) + static_cast<int64_t>(i);
        }
        pos_ids->load(pos_data.data());

        // Embedding lookup
        auto token_tensor = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device);
        token_tensor->load(token_ids);
        auto hidden = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
        ops::embedding(hidden, token_tensor, in_embed);

        // Process each layer
        for (size_t layer = 0; layer < meta.nlayer; layer++) {
            // Attention norm
            auto normed = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
            ops::rms_norm(normed, hidden, attn_norm_w[layer], meta.epsilon);

            // Q, K, V projections
            auto q = Tensor::create({seq_len, meta.nh * meta.dh}, meta.dtype, device);
            auto k = Tensor::create({seq_len, meta.nkvh * meta.dh}, meta.dtype, device);
            auto v = Tensor::create({seq_len, meta.nkvh * meta.dh}, meta.dtype, device);

            ops::linear(q, normed, attn_q_w[layer], attn_q_b[layer]);
            ops::linear(k, normed, attn_k_w[layer], attn_k_b[layer]);
            ops::linear(v, normed, attn_v_w[layer], attn_v_b[layer]);

            // Reshape to [seq_len, n_heads, head_dim]
            auto q_shaped = q->view({seq_len, meta.nh, meta.dh});
            auto k_shaped = k->view({seq_len, meta.nkvh, meta.dh});
            auto v_shaped = v->view({seq_len, meta.nkvh, meta.dh});

            // Apply RoPE
            auto q_rope = Tensor::create({seq_len, meta.nh, meta.dh}, meta.dtype, device);
            auto k_rope = Tensor::create({seq_len, meta.nkvh, meta.dh}, meta.dtype, device);
            ops::rope(q_rope, q_shaped, pos_ids, meta.theta);
            ops::rope(k_rope, k_shaped, pos_ids, meta.theta);

            // Update KV cache
            for (size_t i = 0; i < seq_len; i++) {
                auto k_slice = k_rope->slice(0, i, i + 1);
                auto v_slice = v_shaped->slice(0, i, i + 1);
                auto k_cache_slice = sess.k_cache[layer]->slice(0, sess.cache_pos + i, sess.cache_pos + i + 1);
                auto v_cache_slice = sess.v_cache[layer]->slice(0, sess.cache_pos + i, sess.cache_pos + i + 1);

                // Flatten to 1D and perform D2D copy
                auto k_flat = k_slice->view({meta.nkvh * meta.dh});
                auto v_flat = v_slice->view({meta.nkvh * meta.dh});
                auto kc_flat = k_cache_slice->view({meta.nkvh * meta.dh});
                auto vc_flat = v_cache_slice->view({meta.nkvh * meta.dh});

                size_t copy_bytes = meta.nkvh * meta.dh * k_flat->elementSize();
                core::context().runtime().api()->memcpy_sync(
                    kc_flat->data(), k_flat->data(), copy_bytes, LLAISYS_MEMCPY_D2D);
                core::context().runtime().api()->memcpy_sync(
                    vc_flat->data(), v_flat->data(), copy_bytes, LLAISYS_MEMCPY_D2D);
            }

            // Get full KV from cache
            size_t total_len = sess.cache_pos + seq_len;
            auto k_full = sess.k_cache[layer]->slice(0, 0, total_len);
            auto v_full = sess.v_cache[layer]->slice(0, 0, total_len);

            // Self attention
            auto attn_out = Tensor::create({seq_len, meta.nh, meta.dh}, meta.dtype, device);
            float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));
            ops::self_attention(attn_out, q_rope, k_full, v_full, scale);

            // Output projection
            auto attn_flat = attn_out->view({seq_len, meta.nh * meta.dh});
            auto attn_proj = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
            ops::linear(attn_proj, attn_flat, attn_o_w[layer], nullptr);

            // Residual connection
            ops::add(hidden, hidden, attn_proj);

            // MLP norm
            auto mlp_normed = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
            ops::rms_norm(mlp_normed, hidden, mlp_norm_w[layer], meta.epsilon);

            // MLP
            auto gate = Tensor::create({seq_len, meta.di}, meta.dtype, device);
            auto up = Tensor::create({seq_len, meta.di}, meta.dtype, device);
            ops::linear(gate, mlp_normed, mlp_gate_w[layer], nullptr);
            ops::linear(up, mlp_normed, mlp_up_w[layer], nullptr);

            auto mlp_out = Tensor::create({seq_len, meta.di}, meta.dtype, device);
            ops::swiglu(mlp_out, gate, up);

            auto mlp_proj = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
            ops::linear(mlp_proj, mlp_out, mlp_down_w[layer], nullptr);

            // Residual connection
            ops::add(hidden, hidden, mlp_proj);
        }

        // Final norm
        auto final_normed = Tensor::create({seq_len, meta.hs}, meta.dtype, device);
        ops::rms_norm(final_normed, hidden, out_norm_w, meta.epsilon);

        // Get last token
        auto last_hidden = final_normed->slice(0, seq_len - 1, seq_len);
        auto last_flat = last_hidden->view({meta.hs});

        // LM head
        auto logits = Tensor::create({1, meta.voc}, meta.dtype, device);
        ops::linear(logits, last_flat->view({1, meta.hs}), out_embed, nullptr);

        // 更新 cache 位置并返回设备端 logits
        sess.cache_pos += seq_len;

        return logits->view({meta.voc});
    }

    // ── Argmax 贪心解码（原行为）──────────────────────────────────────────────
    int64_t infer(Qwen2Session &sess, int64_t *token_ids, size_t ntoken) {
        auto logits_flat = run_forward(sess, token_ids, ntoken);
        auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device);
        auto max_val = Tensor::create({1}, LLAISYS_DTYPE_F32, device);
        ops::argmax(max_idx, max_val, logits_flat);
        std::vector<int64_t> result_vec(1);
        core::context().runtime().api()->memcpy_sync(
            reinterpret_cast<std::byte *>(result_vec.data()),
            max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
        return result_vec[0];
    }

    // ── 设备端 logits (BF16/F16/F32) → CPU float32 ───────────────────────────
    void logits_to_f32_cpu(const tensor_t &logits_dev, std::vector<float> &out) {
        out.resize(meta.voc);
        if (meta.dtype == LLAISYS_DTYPE_F32) {
            core::context().runtime().api()->memcpy_sync(
                reinterpret_cast<std::byte *>(out.data()),
                logits_dev->data(), meta.voc * sizeof(float), LLAISYS_MEMCPY_D2H);
        } else {
            std::vector<uint16_t> raw(meta.voc);
            core::context().runtime().api()->memcpy_sync(
                reinterpret_cast<std::byte *>(raw.data()),
                logits_dev->data(), meta.voc * sizeof(uint16_t), LLAISYS_MEMCPY_D2H);
            if (meta.dtype == LLAISYS_DTYPE_BF16) {
                for (size_t i = 0; i < meta.voc; i++) {
                    out[i] = bf16_to_f32(raw[i]);
                }
            } else {
                for (size_t i = 0; i < meta.voc; i++) {
                    out[i] = f16_to_f32(raw[i]);
                }
            }
        }
    }

    // ── Temperature / Top-K / Top-P 采样（CPU 端）───────────────────────────
    int64_t sample_token(std::mt19937 &rng, const std::vector<float> &logits,
                         float temperature, int top_k, float top_p) {
        size_t voc = logits.size();
        // 贪心 argmax
        if (temperature <= 0.0f || top_k == 1) {
            return static_cast<int64_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
        // 带温度的 Softmax
        float max_l = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probs(voc);
        float sum = 0.0f;
        for (size_t i = 0; i < voc; i++) {
            probs[i] = std::exp((logits[i] - max_l) / temperature);
            sum += probs[i];
        }
        for (auto &p : probs) {
            p /= sum;
        }
        // Top-K 截断
        int k = (top_k > 0 && top_k < static_cast<int>(voc)) ? top_k : static_cast<int>(voc);
        std::vector<size_t> idx(voc);
        std::iota(idx.begin(), idx.end(), 0u);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](size_t a, size_t b) { return probs[a] > probs[b]; });
        // Top-P Nucleus 截断
        size_t cutoff = static_cast<size_t>(k);
        if (top_p > 0.0f && top_p < 1.0f) {
            float cum = 0.0f;
            for (int i = 0; i < k; i++) {
                cum += probs[idx[i]];
                if (cum >= top_p) {
                    cutoff = static_cast<size_t>(i) + 1;
                    break;
                }
            }
        }
        // 重归一化后采样
        sum = 0.0f;
        for (size_t i = 0; i < cutoff; i++) {
            sum += probs[idx[i]];
        }
        std::uniform_real_distribution<float> uni(0.0f, sum);
        float r = uni(rng);
        float cum = 0.0f;
        for (size_t i = 0; i < cutoff; i++) {
            cum += probs[idx[i]];
            if (r < cum) {
                return static_cast<int64_t>(idx[i]);
            }
        }
        return static_cast<int64_t>(idx[cutoff - 1]);
    }

    // ── 采样解码 ─────────────────────────────────────────────────────────────
    int64_t infer_sample(Qwen2Session &sess, int64_t *token_ids, size_t ntoken,
                         float temperature, int top_k, float top_p) {
        auto logits_flat = run_forward(sess, token_ids, ntoken);
        std::vector<float> logits_cpu;
        logits_to_f32_cpu(logits_flat, logits_cpu);
        return sample_token(sess.rng, logits_cpu, temperature, top_k, top_p);
    }
};

} // namespace llaisys

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    auto model = new llaisys::Qwen2Model(*meta, device);
    return reinterpret_cast<LlaisysQwen2Model *>(model);
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    delete reinterpret_cast<llaisys::Qwen2Model *>(model);
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model_) {
    auto model = reinterpret_cast<llaisys::Qwen2Model *>(model_);
    auto weights = new LlaisysQwen2Weights();

    weights->in_embed = new LlaisysTensor{model->in_embed};
    weights->out_embed = new LlaisysTensor{model->out_embed};
    weights->out_norm_w = new LlaisysTensor{model->out_norm_w};

    weights->attn_norm_w = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_q_w = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_q_b = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_k_w = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_k_b = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_v_w = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_v_b = new llaisysTensor_t[model->meta.nlayer];
    weights->attn_o_w = new llaisysTensor_t[model->meta.nlayer];
    weights->mlp_norm_w = new llaisysTensor_t[model->meta.nlayer];
    weights->mlp_gate_w = new llaisysTensor_t[model->meta.nlayer];
    weights->mlp_up_w = new llaisysTensor_t[model->meta.nlayer];
    weights->mlp_down_w = new llaisysTensor_t[model->meta.nlayer];

    for (size_t i = 0; i < model->meta.nlayer; i++) {
        weights->attn_norm_w[i] = new LlaisysTensor{model->attn_norm_w[i]};
        weights->attn_q_w[i] = new LlaisysTensor{model->attn_q_w[i]};
        weights->attn_q_b[i] = new LlaisysTensor{model->attn_q_b[i]};
        weights->attn_k_w[i] = new LlaisysTensor{model->attn_k_w[i]};
        weights->attn_k_b[i] = new LlaisysTensor{model->attn_k_b[i]};
        weights->attn_v_w[i] = new LlaisysTensor{model->attn_v_w[i]};
        weights->attn_v_b[i] = new LlaisysTensor{model->attn_v_b[i]};
        weights->attn_o_w[i] = new LlaisysTensor{model->attn_o_w[i]};
        weights->mlp_norm_w[i] = new LlaisysTensor{model->mlp_norm_w[i]};
        weights->mlp_gate_w[i] = new LlaisysTensor{model->mlp_gate_w[i]};
        weights->mlp_up_w[i] = new LlaisysTensor{model->mlp_up_w[i]};
        weights->mlp_down_w[i] = new LlaisysTensor{model->mlp_down_w[i]};
    }

    return weights;
}

// ─── 向后兼容接口：路由到模型内置的 default_session ──────────────────────────

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model_, int64_t *token_ids, size_t ntoken) {
    auto model = reinterpret_cast<llaisys::Qwen2Model *>(model_);
    return model->infer(*model->default_session, token_ids, ntoken);
}

int64_t llaisysQwen2ModelInferSample(struct LlaisysQwen2Model *model_,
                                     int64_t *token_ids, size_t ntoken,
                                     float temperature, int top_k, float top_p) {
    auto model = reinterpret_cast<llaisys::Qwen2Model *>(model_);
    return model->infer_sample(*model->default_session, token_ids, ntoken,
                               temperature, top_k, top_p);
}

void llaisysQwen2ModelSetCachePos(struct LlaisysQwen2Model *model_, size_t pos) {
    reinterpret_cast<llaisys::Qwen2Model *>(model_)->default_session->set_pos(pos);
}

size_t llaisysQwen2ModelGetCachePos(struct LlaisysQwen2Model *model_) {
    return reinterpret_cast<llaisys::Qwen2Model *>(model_)->default_session->get_pos();
}

void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model *model_) {
    reinterpret_cast<llaisys::Qwen2Model *>(model_)->default_session->reset();
}

// ─── 多用户 Session API ───────────────────────────────────────────────────────

struct LlaisysQwen2Session *llaisysQwen2SessionCreate(struct LlaisysQwen2Model *model_) {
    auto model = reinterpret_cast<llaisys::Qwen2Model *>(model_);
    auto sess = new llaisys::Qwen2Session(model->meta, model->device);
    return reinterpret_cast<LlaisysQwen2Session *>(sess);
}

void llaisysQwen2SessionDestroy(struct LlaisysQwen2Session *sess_) {
    delete reinterpret_cast<llaisys::Qwen2Session *>(sess_);
}

int64_t llaisysQwen2SessionInfer(struct LlaisysQwen2Model *model_,
                                 struct LlaisysQwen2Session *sess_,
                                 int64_t *token_ids, size_t ntoken) {
    auto model = reinterpret_cast<llaisys::Qwen2Model *>(model_);
    auto sess = reinterpret_cast<llaisys::Qwen2Session *>(sess_);
    return model->infer(*sess, token_ids, ntoken);
}

int64_t llaisysQwen2SessionInferSample(struct LlaisysQwen2Model *model_,
                                       struct LlaisysQwen2Session *sess_,
                                       int64_t *token_ids, size_t ntoken,
                                       float temperature, int top_k, float top_p) {
    auto model = reinterpret_cast<llaisys::Qwen2Model *>(model_);
    auto sess = reinterpret_cast<llaisys::Qwen2Session *>(sess_);
    return model->infer_sample(*sess, token_ids, ntoken, temperature, top_k, top_p);
}

void llaisysQwen2SessionSetCachePos(struct LlaisysQwen2Session *sess_, size_t pos) {
    reinterpret_cast<llaisys::Qwen2Session *>(sess_)->set_pos(pos);
}

size_t llaisysQwen2SessionGetCachePos(struct LlaisysQwen2Session *sess_) {
    return reinterpret_cast<llaisys::Qwen2Session *>(sess_)->get_pos();
}

void llaisysQwen2SessionResetCache(struct LlaisysQwen2Session *sess_) {
    reinterpret_cast<llaisys::Qwen2Session *>(sess_)->reset();
}

} // extern "C"
