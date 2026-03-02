#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2/qwen2_model.hpp"
#include "../models/qwen2/qwen2_session.hpp"
#include "../utils.hpp"

struct LlaisysQwen2Model {
    llaisys::models::qwen2::Qwen2Model *impl = nullptr;
    LlaisysQwen2Meta meta{};
    LlaisysQwen2Weights weights{};
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;
};

struct LlaisysQwen2Session {
    llaisys::models::qwen2::Qwen2Session *impl = nullptr;
};

namespace {
void log_c_api_error(const char *func, const char *file, int line, const char *message) {
    std::cerr << "[ERROR] " << message << " from " << func << " at " << file << ":" << line << "."
              << std::endl;
}

void init_layer_arrays(LlaisysQwen2Weights &weights, size_t nlayer) {
    weights.in_embed = nullptr;
    weights.out_embed = nullptr;
    weights.out_norm_w = nullptr;

    weights.attn_norm_w = new llaisysTensor_t[nlayer]();
    weights.attn_q_w = new llaisysTensor_t[nlayer]();
    weights.attn_q_b = new llaisysTensor_t[nlayer]();
    weights.attn_k_w = new llaisysTensor_t[nlayer]();
    weights.attn_k_b = new llaisysTensor_t[nlayer]();
    weights.attn_v_w = new llaisysTensor_t[nlayer]();
    weights.attn_v_b = new llaisysTensor_t[nlayer]();
    weights.attn_o_w = new llaisysTensor_t[nlayer]();

    weights.mlp_norm_w = new llaisysTensor_t[nlayer]();
    weights.mlp_gate_w = new llaisysTensor_t[nlayer]();
    weights.mlp_up_w = new llaisysTensor_t[nlayer]();
    weights.mlp_down_w = new llaisysTensor_t[nlayer]();
}

void free_layer_arrays(LlaisysQwen2Weights &weights) {
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

    weights.attn_norm_w = nullptr;
    weights.attn_q_w = nullptr;
    weights.attn_q_b = nullptr;
    weights.attn_k_w = nullptr;
    weights.attn_k_b = nullptr;
    weights.attn_v_w = nullptr;
    weights.attn_v_b = nullptr;
    weights.attn_o_w = nullptr;

    weights.mlp_norm_w = nullptr;
    weights.mlp_gate_w = nullptr;
    weights.mlp_up_w = nullptr;
    weights.mlp_down_w = nullptr;
}
} // namespace

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        if (!meta) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Invalid argument: meta is null");
            return nullptr;
        }

        LlaisysQwen2Model *model = nullptr;
        try {
            model = new LlaisysQwen2Model();
            model->meta = *meta;
            model->device = device;
            model->device_id = (device_ids && ndevice > 0) ? device_ids[0] : 0;

            init_layer_arrays(model->weights, model->meta.nlayer);
            model->impl = new llaisys::models::qwen2::Qwen2Model(model->meta, device, model->device_id);

            return model;
        } catch (const std::exception &e) {
            log_c_api_error(__func__, __FILE__, __LINE__, e.what());
        } catch (...) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Unknown exception");
        }

        if (model) {
            delete model->impl;
            model->impl = nullptr;
            free_layer_arrays(model->weights);
            delete model;
        }
        return nullptr;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        if (!model) {
            return;
        }
        try {
            delete model->impl;
            model->impl = nullptr;
            free_layer_arrays(model->weights);
            delete model;
        } catch (const std::exception &e) {
            log_c_api_error(__func__, __FILE__, __LINE__, e.what());
        } catch (...) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Unknown exception");
        }
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        if (!model) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Invalid argument: model is null");
            return nullptr;
        }
        try {
            return &model->weights;
        } catch (const std::exception &e) {
            log_c_api_error(__func__, __FILE__, __LINE__, e.what());
        } catch (...) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Unknown exception");
        }
        return nullptr;
    }

    struct LlaisysQwen2Session *llaisysQwen2CreateSession(struct LlaisysQwen2Model *model) {
        if (!model) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Invalid argument: model is null");
            return nullptr;
        }
        LlaisysQwen2Session *session = nullptr;
        try {
            session = new LlaisysQwen2Session();
            session->impl = model->impl->create_session();
            return session;
        } catch (const std::exception &e) {
            log_c_api_error(__func__, __FILE__, __LINE__, e.what());
        } catch (...) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Unknown exception");
        }
        if (session) {
            delete session;
        }
        return nullptr;
    }

    void llaisysQwen2DestroySession(struct LlaisysQwen2Session *session) {
        if (!session) {
            return;
        }
        try {
            delete session->impl;
            delete session;
        } catch (const std::exception &e) {
            log_c_api_error(__func__, __FILE__, __LINE__, e.what());
        } catch (...) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Unknown exception");
        }
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, struct LlaisysQwen2Session *session, int64_t *token_ids, size_t ntoken) {
        if (!model) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Invalid argument: model is null");
            return -1;
        }
        if (!session) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Invalid argument: session is null");
            return -1;
        }
        try {
            model->impl->bind_weights(model->weights);
            // Default to argmax for backward compatibility
            return model->impl->infer(session->impl, token_ids, ntoken, 1, 0.0f, 1.0f);
        } catch (const std::exception &e) {
            log_c_api_error(__func__, __FILE__, __LINE__, e.what());
        } catch (...) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Unknown exception");
        }
        return model->meta.end_token;
    }

    int64_t llaisysQwen2ModelInferEx(struct LlaisysQwen2Model *model, struct LlaisysQwen2Session *session, int64_t *token_ids, size_t ntoken, int top_k, float top_p, float temperature, int64_t seed) {
        if (!model) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Invalid argument: model is null");
            return -1;
        }
        if (!session) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Invalid argument: session is null");
            return -1;
        }
        try {
            model->impl->bind_weights(model->weights);
            return model->impl->infer(session->impl, token_ids, ntoken, top_k, top_p, temperature, seed);
        } catch (const std::exception &e) {
            log_c_api_error(__func__, __FILE__, __LINE__, e.what());
        } catch (...) {
            log_c_api_error(__func__, __FILE__, __LINE__, "Unknown exception");
        }
        return model->meta.end_token;
    }
}
