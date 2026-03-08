#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../core/llaisys_core.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/sampling/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace {
using llaisys::tensor_t;

llaisysTensor_t create_tensor_handle(const std::vector<size_t> &shape,
                                     llaisysDataType_t dtype,
                                     llaisysDeviceType_t device,
                                     int device_id) {
    auto t = llaisys::Tensor::create(shape, dtype, device, device_id);
    return new LlaisysTensor{t};
}

void zero_tensor(llaisysTensor_t t) {
    const size_t bytes = t->tensor->numel() * t->tensor->elementSize();
    if (t->tensor->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memset(t->tensor->data(), 0, bytes);
        return;
    }

    auto &ctx = llaisys::core::context();
    ctx.setDevice(t->tensor->deviceType(), t->tensor->deviceId());
    const auto *api = ctx.runtime().api();
    std::vector<std::byte> zeros(bytes, std::byte{0});
    api->memcpy_sync(t->tensor->data(), zeros.data(), bytes, LLAISYS_MEMCPY_H2D);
}

void tensor_write_i64(tensor_t t, int64_t value) {
    if (t->deviceType() == LLAISYS_DEVICE_CPU) {
        *reinterpret_cast<int64_t *>(t->data()) = value;
        return;
    }

    auto &ctx = llaisys::core::context();
    ctx.setDevice(t->deviceType(), t->deviceId());
    const auto *api = ctx.runtime().api();
    api->memcpy_sync(t->data(), &value, sizeof(int64_t), LLAISYS_MEMCPY_H2D);
}

int64_t tensor_read_i64(tensor_t t) {
    if (t->deviceType() == LLAISYS_DEVICE_CPU) {
        return *reinterpret_cast<int64_t *>(t->data());
    }

    auto &ctx = llaisys::core::context();
    ctx.setDevice(t->deviceType(), t->deviceId());
    const auto *api = ctx.runtime().api();
    int64_t value = 0;
    api->memcpy_sync(&value, t->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    return value;
}

void tensor_copy_bytes(tensor_t dst, size_t dst_offset, tensor_t src, size_t src_offset, size_t bytes) {
    auto *dst_ptr = dst->data() + dst_offset;
    auto *src_ptr = src->data() + src_offset;
    if (dst->deviceType() == LLAISYS_DEVICE_CPU && src->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(dst_ptr, src_ptr, bytes);
        return;
    }

    auto &ctx = llaisys::core::context();
    ctx.setDevice(dst->deviceType(), dst->deviceId());
    const auto *api = ctx.runtime().api();
    api->memcpy_sync(dst_ptr, src_ptr, bytes, LLAISYS_MEMCPY_D2D);
}

struct Qwen2ModelImpl {
    LlaisysQwen2Meta meta{};
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;

    LlaisysQwen2Weights weights{};

    // Zero biases (not in weights struct)
    llaisysTensor_t attn_o_b = nullptr;
    llaisysTensor_t mlp_gate_b = nullptr;
    llaisysTensor_t mlp_up_b = nullptr;
    llaisysTensor_t mlp_down_b = nullptr;
    llaisysTensor_t out_b = nullptr;

    // KV cache: per layer
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;

    size_t cur_pos = 0;

    Qwen2ModelImpl(const LlaisysQwen2Meta &m, llaisysDeviceType_t dev, int dev_id)
        : meta(m), device(dev), device_id(dev_id) {}

    void init_weights() {
        // Global weights
        weights.in_embed = create_tensor_handle({meta.voc, meta.hs}, meta.dtype, device, device_id);
        weights.out_embed = create_tensor_handle({meta.voc, meta.hs}, meta.dtype, device, device_id);
        weights.out_norm_w = create_tensor_handle({meta.hs}, meta.dtype, device, device_id);

        // Per-layer weights
        weights.attn_norm_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_q_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_q_b = new llaisysTensor_t[meta.nlayer];
        weights.attn_k_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_k_b = new llaisysTensor_t[meta.nlayer];
        weights.attn_v_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_v_b = new llaisysTensor_t[meta.nlayer];
        weights.attn_o_w = new llaisysTensor_t[meta.nlayer];
        weights.mlp_norm_w = new llaisysTensor_t[meta.nlayer];
        weights.mlp_gate_w = new llaisysTensor_t[meta.nlayer];
        weights.mlp_up_w = new llaisysTensor_t[meta.nlayer];
        weights.mlp_down_w = new llaisysTensor_t[meta.nlayer];

        for (size_t i = 0; i < meta.nlayer; ++i) {
            weights.attn_norm_w[i] = create_tensor_handle({meta.hs}, meta.dtype, device, device_id);
            weights.attn_q_w[i] = create_tensor_handle({meta.nh * meta.dh, meta.hs}, meta.dtype, device, device_id);
            weights.attn_q_b[i] = create_tensor_handle({meta.nh * meta.dh}, meta.dtype, device, device_id);
            weights.attn_k_w[i] = create_tensor_handle({meta.nkvh * meta.dh, meta.hs}, meta.dtype, device, device_id);
            weights.attn_k_b[i] = create_tensor_handle({meta.nkvh * meta.dh}, meta.dtype, device, device_id);
            weights.attn_v_w[i] = create_tensor_handle({meta.nkvh * meta.dh, meta.hs}, meta.dtype, device, device_id);
            weights.attn_v_b[i] = create_tensor_handle({meta.nkvh * meta.dh}, meta.dtype, device, device_id);
            weights.attn_o_w[i] = create_tensor_handle({meta.hs, meta.nh * meta.dh}, meta.dtype, device, device_id);
            weights.mlp_norm_w[i] = create_tensor_handle({meta.hs}, meta.dtype, device, device_id);
            weights.mlp_gate_w[i] = create_tensor_handle({meta.di, meta.hs}, meta.dtype, device, device_id);
            weights.mlp_up_w[i] = create_tensor_handle({meta.di, meta.hs}, meta.dtype, device, device_id);
            weights.mlp_down_w[i] = create_tensor_handle({meta.hs, meta.di}, meta.dtype, device, device_id);

            zero_tensor(weights.attn_q_b[i]);
            zero_tensor(weights.attn_k_b[i]);
            zero_tensor(weights.attn_v_b[i]);
        }

        // Extra zero biases
        attn_o_b = create_tensor_handle({meta.hs}, meta.dtype, device, device_id);
        mlp_gate_b = create_tensor_handle({meta.di}, meta.dtype, device, device_id);
        mlp_up_b = create_tensor_handle({meta.di}, meta.dtype, device, device_id);
        mlp_down_b = create_tensor_handle({meta.hs}, meta.dtype, device, device_id);
        out_b = create_tensor_handle({meta.voc}, meta.dtype, device, device_id);

        zero_tensor(attn_o_b);
        zero_tensor(mlp_gate_b);
        zero_tensor(mlp_up_b);
        zero_tensor(mlp_down_b);
        zero_tensor(out_b);

        // KV cache
        k_cache.resize(meta.nlayer);
        v_cache.resize(meta.nlayer);
        for (size_t i = 0; i < meta.nlayer; ++i) {
            k_cache[i] = llaisys::Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device, device_id);
            v_cache[i] = llaisys::Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device, device_id);
            if (device == LLAISYS_DEVICE_CPU) {
                std::memset(k_cache[i]->data(), 0, k_cache[i]->numel() * k_cache[i]->elementSize());
                std::memset(v_cache[i]->data(), 0, v_cache[i]->numel() * v_cache[i]->elementSize());
            } else {
                auto k_handle = new LlaisysTensor{k_cache[i]};
                auto v_handle = new LlaisysTensor{v_cache[i]};
                zero_tensor(k_handle);
                zero_tensor(v_handle);
                delete k_handle;
                delete v_handle;
            }
        }
    }

    void destroy_weights() {
        delete weights.in_embed;
        delete weights.out_embed;
        delete weights.out_norm_w;

        for (size_t i = 0; i < meta.nlayer; ++i) {
            delete weights.attn_norm_w[i];
            delete weights.attn_q_w[i];
            delete weights.attn_q_b[i];
            delete weights.attn_k_w[i];
            delete weights.attn_k_b[i];
            delete weights.attn_v_w[i];
            delete weights.attn_v_b[i];
            delete weights.attn_o_w[i];
            delete weights.mlp_norm_w[i];
            delete weights.mlp_gate_w[i];
            delete weights.mlp_up_w[i];
            delete weights.mlp_down_w[i];
        }

        delete[] weights.attn_norm_w;
        delete[] weights.attn_q_w;
        delete[] weights.attn_q_b;
        delete[] weights.attn_k_w;
        delete[] weights.attn_k_b;
        delete[] weights.attn_v_w;
        delete[] weights.attn_v_b;
        delete[] weights.attn_o_w;
        delete[] weights.mlp_norm_w;
        delete[] weights.mlp_gate_w;
        delete[] weights.mlp_up_w;
        delete[] weights.mlp_down_w;

        delete attn_o_b;
        delete mlp_gate_b;
        delete mlp_up_b;
        delete mlp_down_b;
        delete out_b;
    }

    int64_t infer_next(const int64_t *token_ids, size_t ntoken, int top_k, float top_p, float temperature) {
        if (ntoken == 0) {
            return meta.end_token;
        }

        // Reset cache if input sequence is shorter than cached position
        if (ntoken < cur_pos) {
            cur_pos = 0;
        }

        int64_t next_token = meta.end_token;

        // Process tokens from cur_pos onwards (KV-Cache optimization)
        for (size_t i = cur_pos; i < ntoken; ++i) {
            int64_t token_id = token_ids[i];

            // Token embedding
            auto token_tensor = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
            tensor_write_i64(token_tensor, token_id);

            auto x = llaisys::Tensor::create({1, meta.hs}, meta.dtype, device, device_id);
            llaisys::ops::embedding(x, token_tensor, weights.in_embed->tensor);

            // Transformer layers
            for (size_t l = 0; l < meta.nlayer; ++l) {
                // Self-Attention block
                auto x_norm = llaisys::Tensor::create({1, meta.hs}, meta.dtype, device, device_id);
                llaisys::ops::rms_norm(x_norm, x, weights.attn_norm_w[l]->tensor, meta.epsilon);

                // Q, K, V projections
                auto q_lin = llaisys::Tensor::create({1, meta.nh * meta.dh}, meta.dtype, device, device_id);
                auto k_lin = llaisys::Tensor::create({1, meta.nkvh * meta.dh}, meta.dtype, device, device_id);
                auto v_lin = llaisys::Tensor::create({1, meta.nkvh * meta.dh}, meta.dtype, device, device_id);

                llaisys::ops::linear(q_lin, x_norm, weights.attn_q_w[l]->tensor, weights.attn_q_b[l]->tensor);
                llaisys::ops::linear(k_lin, x_norm, weights.attn_k_w[l]->tensor, weights.attn_k_b[l]->tensor);
                llaisys::ops::linear(v_lin, x_norm, weights.attn_v_w[l]->tensor, weights.attn_v_b[l]->tensor);

                auto q = q_lin->view({1, meta.nh, meta.dh});
                auto k = k_lin->view({1, meta.nkvh, meta.dh});
                auto v = v_lin->view({1, meta.nkvh, meta.dh});

                // RoPE
                auto pos_ids = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
                tensor_write_i64(pos_ids, static_cast<int64_t>(i));

                auto q_rot = llaisys::Tensor::create({1, meta.nh, meta.dh}, meta.dtype, device, device_id);
                auto k_rot = llaisys::Tensor::create({1, meta.nkvh, meta.dh}, meta.dtype, device, device_id);

                llaisys::ops::rope(q_rot, q, pos_ids, meta.theta);
                llaisys::ops::rope(k_rot, k, pos_ids, meta.theta);

                // Update KV cache
                size_t bytes_per_token = meta.nkvh * meta.dh * x->elementSize();
                tensor_copy_bytes(k_cache[l], i * bytes_per_token, k_rot, 0, bytes_per_token);
                tensor_copy_bytes(v_cache[l], i * bytes_per_token, v, 0, bytes_per_token);

                // Get all cached K, V for attention
                auto k_all = k_cache[l]->slice(0, 0, i + 1);
                auto v_all = v_cache[l]->slice(0, 0, i + 1);

                // Self-attention
                auto attn_out = llaisys::Tensor::create({1, meta.nh, meta.dh}, meta.dtype, device, device_id);
                float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));
                llaisys::ops::self_attention(attn_out, q_rot, k_all, v_all, scale);

                // Output projection
                auto attn_out_2d = attn_out->view({1, meta.nh * meta.dh});
                auto attn_proj = llaisys::Tensor::create({1, meta.hs}, meta.dtype, device, device_id);
                llaisys::ops::linear(attn_proj, attn_out_2d, weights.attn_o_w[l]->tensor, attn_o_b->tensor);

                // Residual connection
                llaisys::ops::add(x, x, attn_proj);

                // MLP block
                auto x_norm2 = llaisys::Tensor::create({1, meta.hs}, meta.dtype, device, device_id);
                llaisys::ops::rms_norm(x_norm2, x, weights.mlp_norm_w[l]->tensor, meta.epsilon);

                auto gate = llaisys::Tensor::create({1, meta.di}, meta.dtype, device, device_id);
                auto up = llaisys::Tensor::create({1, meta.di}, meta.dtype, device, device_id);
                llaisys::ops::linear(gate, x_norm2, weights.mlp_gate_w[l]->tensor, mlp_gate_b->tensor);
                llaisys::ops::linear(up, x_norm2, weights.mlp_up_w[l]->tensor, mlp_up_b->tensor);

                auto swiglu_out = llaisys::Tensor::create({1, meta.di}, meta.dtype, device, device_id);
                llaisys::ops::swiglu(swiglu_out, gate, up);

                auto mlp_out = llaisys::Tensor::create({1, meta.hs}, meta.dtype, device, device_id);
                llaisys::ops::linear(mlp_out, swiglu_out, weights.mlp_down_w[l]->tensor, mlp_down_b->tensor);

                // Residual connection
                llaisys::ops::add(x, x, mlp_out);
            }

            // Final layer norm
            auto x_norm_final = llaisys::Tensor::create({1, meta.hs}, meta.dtype, device, device_id);
            llaisys::ops::rms_norm(x_norm_final, x, weights.out_norm_w->tensor, meta.epsilon);

            // Output logits
            auto logits = llaisys::Tensor::create({1, meta.voc}, meta.dtype, device, device_id);
            llaisys::ops::linear(logits, x_norm_final, weights.out_embed->tensor, out_b->tensor);

            // Sampling for next token (argmax when top_k<=1 or temperature<=0)
            auto logits_1d = logits->view({meta.voc});
            auto sampled_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
            llaisys::ops::sample(sampled_idx, logits_1d, temperature, top_k, top_p);
            next_token = tensor_read_i64(sampled_idx);
        }

        cur_pos = ntoken;
        return next_token;
    }
};
} // namespace

struct LlaisysQwen2Model {
    Qwen2ModelImpl *impl;
};

__C {

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                      llaisysDeviceType_t device,
                                                      int *device_ids,
                                                      int ndevice) {
        try {
            CHECK_ARGUMENT(meta != nullptr, "Qwen2: meta must not be null.");
            CHECK_ARGUMENT(ndevice >= 1, "Qwen2: must have at least one device.");
            int device_id = device_ids ? device_ids[0] : 0;
            auto *model = new LlaisysQwen2Model();
            model->impl = new Qwen2ModelImpl(*meta, device, device_id);
            model->impl->init_weights();
            return model;
        } catch (...) {
            return nullptr;
        }
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        try {
            if (!model) {
                return;
            }
            model->impl->destroy_weights();
            delete model->impl;
            delete model;
        } catch (...) {
            return;
        }
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2: model must not be null.");
            return &model->impl->weights;
        } catch (...) {
            return nullptr;
        }
    }

    llaisysTensor_t llaisysQwen2ModelGetWeight(struct LlaisysQwen2Model * model, int kind, size_t layer) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2: model must not be null.");
            auto &w = model->impl->weights;
            switch (kind) {
            case LLAISYS_QWEN2_WEIGHT_IN_EMBED:
                return w.in_embed;
            case LLAISYS_QWEN2_WEIGHT_OUT_EMBED:
                return w.out_embed;
            case LLAISYS_QWEN2_WEIGHT_OUT_NORM:
                return w.out_norm_w;
            case LLAISYS_QWEN2_WEIGHT_ATTN_NORM:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.attn_norm_w[layer];
            case LLAISYS_QWEN2_WEIGHT_ATTN_Q_W:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.attn_q_w[layer];
            case LLAISYS_QWEN2_WEIGHT_ATTN_Q_B:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.attn_q_b[layer];
            case LLAISYS_QWEN2_WEIGHT_ATTN_K_W:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.attn_k_w[layer];
            case LLAISYS_QWEN2_WEIGHT_ATTN_K_B:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.attn_k_b[layer];
            case LLAISYS_QWEN2_WEIGHT_ATTN_V_W:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.attn_v_w[layer];
            case LLAISYS_QWEN2_WEIGHT_ATTN_V_B:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.attn_v_b[layer];
            case LLAISYS_QWEN2_WEIGHT_ATTN_O_W:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.attn_o_w[layer];
            case LLAISYS_QWEN2_WEIGHT_MLP_NORM:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.mlp_norm_w[layer];
            case LLAISYS_QWEN2_WEIGHT_MLP_GATE_W:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.mlp_gate_w[layer];
            case LLAISYS_QWEN2_WEIGHT_MLP_UP_W:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.mlp_up_w[layer];
            case LLAISYS_QWEN2_WEIGHT_MLP_DOWN_W:
                CHECK_ARGUMENT(layer < model->impl->meta.nlayer, "Qwen2: layer out of range.");
                return w.mlp_down_w[layer];
            default:
                return nullptr;
            }
        } catch (...) {
            return nullptr;
        }
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model,
                                   int64_t *token_ids,
                                   size_t ntoken,
                                   int top_k,
                                   float top_p,
                                   float temperature) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2: model must not be null.");
            CHECK_ARGUMENT(token_ids != nullptr, "Qwen2: token_ids must not be null.");
            return model->impl->infer_next(token_ids, ntoken, top_k, top_p, temperature);
        } catch (...) {
            return -1;
        }
    }
}
