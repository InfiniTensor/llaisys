#include "llaisys/models/qwen2.h"
#include "../kvcache/simple.hpp"
#include "../ops/ops.hpp"
#include "llaisys.h"
#include "llaisys/ops.h"
#include "llaisys/tensor.h"
#include "llaisys_tensor.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

#define DBG_LOG false

#define createTensor(...)                                                      \
    new LlaisysTensor { llaisys::Tensor::create(__VA_ARGS__) }

#define loadTensor(ts)                                                         \
    [&]() {                                                                    \
        auto t = new LlaisysTensor{ts};                                        \
        return t;                                                              \
    }()

#define CASE(id, name, val)                                                    \
    case id:                                                                   \
        do {                                                                   \
            if (layer_id != -1) {                                              \
                std::cerr << "[qwen2.cc:setWeights()] " #name                  \
                             " should not have layer_id"                       \
                          << std::endl;                                        \
                exit(1);                                                       \
            }                                                                  \
            auto ts = val->tensor;                                             \
            model->weights.name = createTensor(                                \
                ts->shape(), ts->dtype(), model->device, model->device_id);    \
            model->weights.name->tensor = ts;                                  \
            if constexpr (DBG_LOG) {                                           \
                std::cerr << "[qwen2.cc:setWeights()] Set " #name              \
                          << std::endl;                                        \
            }                                                                  \
        } while (0);                                                           \
        break;

#define CASE_ARRAY(id, name, val)                                              \
    case id:                                                                   \
        do {                                                                   \
            if (layer_id < 0 || layer_id >= static_cast<int>(nlayer)) {        \
                std::cerr << "[qwen2.cc:setWeights()] " #name                  \
                             " layer_id out of range"                          \
                          << std::endl;                                        \
                exit(1);                                                       \
            }                                                                  \
            auto ts = val->tensor;                                             \
            model->weights.name[layer_id] = createTensor(                      \
                ts->shape(), ts->dtype(), model->device, model->device_id);    \
            model->weights.name[layer_id]->tensor = ts;                        \
            if constexpr (DBG_LOG) {                                           \
                std::cerr << "[qwen2.cc:setWeights()] Set " #name              \
                          << " for layer " << layer_id << std::endl;           \
            }                                                                  \
        } while (0);                                                           \
        break;

#define MODEL_VALIDITY_CHECK(model)                                            \
    do {                                                                       \
        if (!model) {                                                          \
            std::cerr << "[qwen2.cc:infer()] Model is null, cannot perform "   \
                         "inference."                                          \
                      << std::endl;                                            \
            return -1;                                                         \
        }                                                                      \
    } while (0)

#define LOG_SHAPE(stage, tensr, name)                                          \
    do {                                                                       \
        if constexpr (!DBG_LOG)                                                \
            break;                                                             \
        std::cerr << "[qwen2.cc:" << stage << "] " << name << " shape: ";      \
        for (int i = 0, l = int(tensr->shape().size()); i < l; ++i) {          \
            std::cerr << tensr->shape()[i];                                    \
            if (i != l - 1)                                                    \
                std::cerr << " x ";                                            \
        }                                                                      \
        std::cerr << std::endl;                                                \
    } while (0)
// Define some helper functions here, init model weights array/kvcache
// array, etc.
static void initializeArrays(LlaisysQwen2Model *model);

static int64_t qwen2_infer_impl(struct LlaisysQwen2Model *model,
                                int64_t *token_ids,
                                int64_t *pos_ids,
                                size_t ntoken,
                                bool prefill,
                                int top_k,
                                float top_p,
                                float temperature);

__C {

    struct LlaisysQwen2Model {
        LlaisysQwen2Meta meta;
        LlaisysQwen2Weights weights;
        llaisys::kvcache::simple::KVCache **kvcaches;
        llaisysDeviceType_t device;
        int device_id;
    };

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, llaisysDeviceType_t device,
        int *device_ids, int ndevice) {
        if (!meta) {
            std::cerr
                << "[qwen2.cc:create()] Meta is null, cannot create model."
                << std::endl;
            return nullptr;
        }

        LlaisysQwen2Model *model = new LlaisysQwen2Model();
        model->meta = *meta;
        model->device = device;
        model->device_id = device_ids ? device_ids[0] : 0;

        initializeArrays(model);
        std::cerr << "[qwen2.cc:create()] Model created." << std::endl;

        return model;
    }

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (!model) {
            std::cerr
                << "[qwen2.cc:destroy()] Model is null, nothing to destroy."
                << std::endl;
            return;
        }

        // Free KV caches
        for (size_t i = 0; i < model->meta.nlayer; ++i) {
            delete model->kvcaches[i];
        }
        delete[] model->kvcaches;
        std::cerr << "[qwen2.cc:destroy()] Destroyed all KV caches."
                  << std::endl;

        // Free weight arrays
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
        std::cerr << "[qwen2.cc:destroy()] Destroyed all weight arrays."
                  << std::endl;

        delete model;
        std::cerr << "[qwen2.cc:destroy()] Model destroyed." << std::endl;
    }

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(
        struct LlaisysQwen2Model * model) {
        if (!model) {
            std::cerr
                << "[qwen2.cc:weights()] Model is null, cannot get weights."
                << std::endl;
            return nullptr;
        }
        /**
         * Display all tensor shape here for debugging
         */
        if constexpr (DBG_LOG)
            std::cerr << "[qwen2.cc:weights()] Model weights shapes:"
                      << std::endl;

        LOG_SHAPE("weights()", model->weights.in_embed->tensor, "in_embed");
        LOG_SHAPE("weights()", model->weights.out_embed->tensor, "out_embed");
        LOG_SHAPE("weights()", model->weights.out_norm_w->tensor, "out_norm_w");

        auto nlayer = model->meta.nlayer;
        for (size_t i = 0; i < nlayer; ++i) {
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:weights()] Layer " << i
                          << " weights:" << std::endl;
            LOG_SHAPE("weights()", model->weights.attn_norm_w[i]->tensor,
                      "attn_norm_w");
            LOG_SHAPE("weights()", model->weights.attn_q_w[i]->tensor,
                      "attn_q_w");
            LOG_SHAPE("weights()", model->weights.attn_q_b[i]->tensor,
                      "attn_q_b");
            LOG_SHAPE("weights()", model->weights.attn_k_w[i]->tensor,
                      "attn_k_w");
            LOG_SHAPE("weights()", model->weights.attn_k_b[i]->tensor,
                      "attn_k_b");
            LOG_SHAPE("weights()", model->weights.attn_v_w[i]->tensor,
                      "attn_v_w");
            LOG_SHAPE("weights()", model->weights.attn_v_b[i]->tensor,
                      "attn_v_b");
            LOG_SHAPE("weights()", model->weights.attn_o_w[i]->tensor,
                      "attn_o_w");
            LOG_SHAPE("weights()", model->weights.mlp_norm_w[i]->tensor,
                      "mlp_norm_w");
            LOG_SHAPE("weights()", model->weights.mlp_gate_w[i]->tensor,
                      "mlp_gate_w");
            LOG_SHAPE("weights()", model->weights.mlp_up_w[i]->tensor,
                      "mlp_up_w");
            LOG_SHAPE("weights()", model->weights.mlp_down_w[i]->tensor,
                      "mlp_down_w");
        }

        return &model->weights;
    }

    __export void llaisysQwen2SetWeights(struct LlaisysQwen2Model * model,
                                         int name, int layer_id,
                                         llaisysTensor_t tensor) {
        if (!model) {
            std::cerr
                << "[qwen2.cc:setWeights()] Model is null, cannot set weights."
                << std::endl;
            return;
        }

        size_t nlayer = model->meta.nlayer;
        switch (name) {
            CASE(0, in_embed, tensor)          // in_embed
            CASE(1, out_embed, tensor)         // out_embed
            CASE(2, out_norm_w, tensor)        // out_norm_w
            CASE_ARRAY(3, attn_norm_w, tensor) // attn_norm_w
            CASE_ARRAY(4, attn_q_w, tensor)    // attn_q_w
            CASE_ARRAY(5, attn_q_b, tensor)    // attn_q_b
            CASE_ARRAY(6, attn_k_w, tensor)    // attn_k_w
            CASE_ARRAY(7, attn_k_b, tensor)    // attn_k_b
            CASE_ARRAY(8, attn_v_w, tensor)    // attn_v_w
            CASE_ARRAY(9, attn_v_b, tensor)    // attn_v_b
            CASE_ARRAY(10, attn_o_w, tensor)   // attn_o_w
            CASE_ARRAY(11, mlp_norm_w, tensor) // mlp_norm_w
            CASE_ARRAY(12, mlp_gate_w, tensor) // mlp_gate_w
            CASE_ARRAY(13, mlp_up_w, tensor)   // mlp_up_w
            CASE_ARRAY(14, mlp_down_w, tensor) // mlp_down_w
        default:
            std::cerr << "[qwen2.cc:setWeights()] Unknown weight name: " << name
                      << ", cannot set weight." << std::endl;
            exit(1);
        }

        LOG_SHAPE("setWeights()", tensor->tensor, "weight tensor from Python");
    }

    __export int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model * model, int64_t *token_ids, int64_t *pos_ids,
        size_t ntoken, bool prefill) {
        // Keep old API behavior: greedy decoding via top_k=1.
        return qwen2_infer_impl(model, token_ids, pos_ids, ntoken, prefill,
                                1, 1.0f, 1.0f);
    }

    __export int64_t llaisysQwen2ModelInferSample(
        struct LlaisysQwen2Model * model, int64_t *token_ids, int64_t *pos_ids,
        size_t ntoken, bool prefill, int top_k, float top_p,
        float temperature) {
        return qwen2_infer_impl(model, token_ids, pos_ids, ntoken, prefill,
                                top_k, top_p, temperature);
    }
}

static int64_t qwen2_infer_impl(struct LlaisysQwen2Model * model,
                                int64_t *token_ids, int64_t *pos_ids,
                                size_t ntoken, bool prefill, int top_k,
                                float top_p, float temperature) {
        //* -1. Do checking
        MODEL_VALIDITY_CHECK(model);
        if (ntoken == 0) {
            std::cerr << "[qwen2.cc:infer()] ntoken must be > 0." << std::endl;
            return -1;
        }
        if (top_k < 0 || top_p <= 0.0f || top_p > 1.0f || temperature <= 0.0f) {
            std::cerr << "[qwen2.cc:infer()] Invalid sampling params: top_k="
                      << top_k << ", top_p=" << top_p
                      << ", temperature=" << temperature << std::endl;
            return -1;
        }
        if constexpr (DBG_LOG)
            std::cerr << "[qwen2.cc:infer()] Start inference." << std::endl;

        //* 0. If prefill, clean KV Caches
        if (prefill) {
            if constexpr (DBG_LOG)
                std::cerr
                    << "[qwen2.cc:infer()] Prefill mode: resetting KV caches."
                    << std::endl;
            for (size_t i = 0; i < model->meta.nlayer; ++i)
                model->kvcaches[i]->reset();
        }

        //* 1. Copy inputs into tensor
        using namespace llaisys;
        using tensor = tensor_t;
        using usize = size_t;
        using i64 = int64_t;

        tensor pos_ids_tensor = Tensor::create({ntoken}, LLAISYS_DTYPE_I64,
                                               model->device, model->device_id);
        pos_ids_tensor->load(pos_ids);
        if constexpr (DBG_LOG)
            std::cerr << "[qwen2.cc:infer()] Loaded position ids." << std::endl;
        LOG_SHAPE("infer()", pos_ids_tensor, "pos_ids");

        tensor input_tokens = Tensor::create({ntoken}, LLAISYS_DTYPE_I64,
                                             model->device, model->device_id);
        input_tokens->load(token_ids);
        if constexpr (DBG_LOG)
            std::cerr << "[qwen2.cc:infer()] Loaded input token ids."
                      << std::endl;
        LOG_SHAPE("infer()", input_tokens, "input_tokens");

        //* 2. Token Embedding
        tensor hidden_states
            = Tensor::create({ntoken, model->meta.hs}, model->meta.dtype,
                             model->device, model->device_id);
        ops::embedding(hidden_states, input_tokens,
                       model->weights.in_embed->tensor);
        if constexpr (DBG_LOG)
            std::cerr << "[qwen2.cc:infer()] Completed token embedding."
                      << std::endl;
        LOG_SHAPE("infer()", hidden_states, "hidden_states");

        //* 3. Attention Layers
        for (usize layer = 0; layer < model->meta.nlayer; ++layer) {
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": (Mock) Completed layer operation." << std::endl;

            //* 3.a Record a residual
            tensor residual = hidden_states;

            //* 3.b RMS Norm before Attention
            tensor attn_normed
                = Tensor::create({ntoken, model->meta.hs}, model->meta.dtype,
                                 model->device, model->device_id);
            ops::rms_norm(attn_normed, hidden_states,
                          model->weights.attn_norm_w[layer]->tensor,
                          model->meta.epsilon);
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Completed RMS norm before attention."
                          << std::endl;

            //* 3.c QKV Projection
            tensor q_proj = Tensor::create(
                {ntoken, model->meta.nh * model->meta.dh}, model->meta.dtype,
                model->device, model->device_id);
            tensor k_proj = Tensor::create(
                {ntoken, model->meta.nkvh * model->meta.dh}, model->meta.dtype,
                model->device, model->device_id);
            tensor v_proj = Tensor::create(
                {ntoken, model->meta.nkvh * model->meta.dh}, model->meta.dtype,
                model->device, model->device_id);
            ops::linear(q_proj, attn_normed,
                        model->weights.attn_q_w[layer]->tensor,
                        model->weights.attn_q_b[layer]->tensor);
            ops::linear(k_proj, attn_normed,
                        model->weights.attn_k_w[layer]->tensor,
                        model->weights.attn_k_b[layer]->tensor);
            ops::linear(v_proj, attn_normed,
                        model->weights.attn_v_w[layer]->tensor,
                        model->weights.attn_v_b[layer]->tensor);
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Completed QKV projection." << std::endl;

            //* 3.c.1 Reshape to (S, H, D)
            tensor qview
                = q_proj->view({ntoken, model->meta.nh, model->meta.dh});
            tensor kview
                = k_proj->view({ntoken, model->meta.nkvh, model->meta.dh});
            tensor vview
                = v_proj->view({ntoken, model->meta.nkvh, model->meta.dh});

            //* 3.d RoPE Encoding for Q, K
            tensor pos_q = Tensor::create(
                {ntoken, model->meta.nh, model->meta.dh}, model->meta.dtype,
                model->device, model->device_id);
            tensor pos_k = Tensor::create(
                {ntoken, model->meta.nkvh, model->meta.dh}, model->meta.dtype,
                model->device, model->device_id);
            ops::rope(pos_q, qview, pos_ids_tensor, model->meta.theta);
            ops::rope(pos_k, kview, pos_ids_tensor, model->meta.theta);
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Completed RoPE encoding for Q and K."
                          << std::endl;

            //* 3.e Update KV Cache
            model->kvcaches[layer]->insert(pos_k, vview, ntoken);

            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Updated KV cache." << std::endl;

            //* 3.f Self attention
            float scale = 1.0f / std::sqrt(static_cast<float>(model->meta.dh));
            tensor kcache = model->kvcaches[layer]->getKeysSlice();
            if (prefill) {
                // ASSERT(kcache->shape() == pos_k->shape(),
                //        "K cache shape mismatch!");
            } else {
                if constexpr (DBG_LOG)
                    std::cerr
                        << "[qwen2.cc:infer()] Decode mode - pos_k shape: ["
                        << pos_k->shape()[0] << ", " << pos_k->shape()[1]
                        << ", " << pos_k->shape()[2] << "], "
                        << "kcache shape: [" << kcache->shape()[0] << ", "
                        << kcache->shape()[1] << ", " << kcache->shape()[2]
                        << "]" << std::endl;
            }
            tensor vcache = model->kvcaches[layer]->getValuesSlice();

            tensor attn_out = Tensor::create(
                {ntoken, model->meta.nh, model->meta.dh}, model->meta.dtype,
                model->device, model->device_id);
            ops::self_attention(attn_out, pos_q, kcache, vcache, scale);
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Completed self-attention computation."
                          << std::endl;

            //* 3.g Output Projection
            tensor attn_proj
                = Tensor::create({ntoken, model->meta.hs}, model->meta.dtype,
                                 model->device, model->device_id);
            ops::linear(
                attn_proj,
                attn_out->view({ntoken, model->meta.nh * model->meta.dh}),
                model->weights.attn_o_w[layer]->tensor, nullptr);
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Completed attention output projection."
                          << std::endl;

            //* 3.h Residual after attention
            ops::add(hidden_states, residual, attn_proj);
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Completed residual addition after attention."
                          << std::endl;
            residual = hidden_states;

            //* 3.i MLP block
            tensor mlp_normed
                = Tensor::create({ntoken, model->meta.hs}, model->meta.dtype,
                                 model->device, model->device_id);
            ops::rms_norm(mlp_normed, hidden_states,
                          model->weights.mlp_norm_w[layer]->tensor,
                          model->meta.epsilon);
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Completed RMS norm before MLP." << std::endl;

            //* 3.j MLP projections
            tensor mlp_gate
                = Tensor::create({ntoken, model->meta.di}, model->meta.dtype,
                                 model->device, model->device_id);
            tensor mlp_up
                = Tensor::create({ntoken, model->meta.di}, model->meta.dtype,
                                 model->device, model->device_id);
            ops::linear(mlp_gate, mlp_normed,
                        model->weights.mlp_gate_w[layer]->tensor, nullptr);
            ops::linear(mlp_up, mlp_normed,
                        model->weights.mlp_up_w[layer]->tensor, nullptr);
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Completed MLP gate and up projections."
                          << std::endl;

            tensor mlp_down
                = Tensor::create({ntoken, model->meta.di}, model->meta.dtype,
                                 model->device, model->device_id);
            ops::swiglu(mlp_down, mlp_gate, mlp_up);
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Completed SwiGLU activation." << std::endl;

            //* 3.k Final MLP output projection
            tensor mlp_out
                = Tensor::create({ntoken, model->meta.hs}, model->meta.dtype,
                                 model->device, model->device_id);
            ops::linear(mlp_out, mlp_down,
                        model->weights.mlp_down_w[layer]->tensor, nullptr);
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Completed final MLP output projection."
                          << std::endl;

            //* 3.l Final residual addition
            ops::add(hidden_states, residual, mlp_out);
            if constexpr (DBG_LOG)
                std::cerr << "[qwen2.cc:infer()] Layer " << layer
                          << ": Completed final residual addition."
                          << std::endl;
        }

        //* 4. Final transform and output projection
        tensor final_norm
            = Tensor::create({ntoken, model->meta.hs}, model->meta.dtype,
                             model->device, model->device_id);
        ops::rms_norm(final_norm, hidden_states,
                      model->weights.out_norm_w->tensor, model->meta.epsilon);
        if constexpr (DBG_LOG)
            std::cerr << "[qwen2.cc:infer()] Completed final RMS norm."
                      << std::endl;

        //* 5. Get logits
        tensor logits
            = Tensor::create({ntoken, model->meta.voc}, model->meta.dtype,
                             model->device, model->device_id);
        ops::linear(logits, final_norm, model->weights.out_embed->tensor,
                    nullptr);
        if constexpr (DBG_LOG)
            std::cerr << "[qwen2.cc:infer()] Completed output projection "
                         "to logits."
                      << std::endl;

        //* 6. Get the last token's logits and argmax
        tensor last_token_logits
            = logits->slice(0, ntoken - 1, ntoken); // last token
        if constexpr (DBG_LOG)
            std::cerr << "[qwen2.cc:infer()] Sliced out last token logits."
                      << std::endl;
        LOG_SHAPE("infer()", logits, "logits");
        LOG_SHAPE("infer()", last_token_logits, "last_token_logits");

        //* 7. Sample to get next token id (top_k=1 equals argmax)
        // sample() currently supports CPU only; move logits to host when needed.
        tensor sample_logits = last_token_logits;
        if (last_token_logits->deviceType() != LLAISYS_DEVICE_CPU) {
            sample_logits = Tensor::create(
                last_token_logits->shape(), last_token_logits->dtype(),
                LLAISYS_DEVICE_CPU, 0);
            llaisys::core::context().setDevice(model->device, model->device_id);
            llaisys::core::context().runtime().api()->memcpy_sync(
                sample_logits->data(), last_token_logits->data(),
                last_token_logits->numel() * last_token_logits->elementSize(),
                LLAISYS_MEMCPY_D2H);
        }

        tensor next_token_id_tensor
            = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        ops::sample(next_token_id_tensor, sample_logits, top_k, top_p,
                    temperature);
        if constexpr (DBG_LOG)
            std::cerr << "[qwen2.cc:infer()] Completed sampling to get next "
                         "token id: "
                      << *((i64 *)next_token_id_tensor->data()) << std::endl;

        // NOTE: keep a safe read path for both host/device output tensors.
        i64 next_token_id = -1;
        if (next_token_id_tensor->deviceType() == LLAISYS_DEVICE_CPU) {
            next_token_id = *((i64 *)next_token_id_tensor->data());
        } else {
            llaisys::core::context().setDevice(model->device, model->device_id);
            llaisys::core::context().runtime().api()->memcpy_sync(
                &next_token_id, next_token_id_tensor->data(), sizeof(i64),
                LLAISYS_MEMCPY_D2H);
        }
    return next_token_id;
}

static void initializeArrays(LlaisysQwen2Model *model) {
    size_t nlayer = model->meta.nlayer;

    // KV Cache init
    model->kvcaches = new llaisys::kvcache::simple::KVCache *[nlayer];
    for (size_t i = 0; i < nlayer; ++i) {
        model->kvcaches[i] = new llaisys::kvcache::simple::KVCache(
            model->meta.maxseq, model->meta.nkvh, model->meta.dh,
            model->meta.dh, model->meta.dtype, model->device, model->device_id);
    }

    std::cerr << "[qwen2.cc:initializeArrays()] Initialized KV caches for "
              << nlayer << " layers." << std::endl;

    // Weight init
    model->weights.attn_norm_w = new llaisysTensor_t[nlayer];
    model->weights.attn_q_w = new llaisysTensor_t[nlayer];
    model->weights.attn_q_b = new llaisysTensor_t[nlayer];
    model->weights.attn_k_w = new llaisysTensor_t[nlayer];
    model->weights.attn_k_b = new llaisysTensor_t[nlayer];
    model->weights.attn_v_w = new llaisysTensor_t[nlayer];
    model->weights.attn_v_b = new llaisysTensor_t[nlayer];
    model->weights.attn_o_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_norm_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_gate_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_up_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_down_w = new llaisysTensor_t[nlayer];

    std::cerr << "[qwen2.cc:initializeArrays()] Initialized Qwen2 model with "
              << nlayer << " layers." << std::endl;
}