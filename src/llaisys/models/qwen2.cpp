// Qwen2 C API implementation (skeleton)
#include "llaisys/models/qwen2.h"
#include "../../models/qwen2/qwen2.hpp"
#include "qwen2_kv_internal.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

struct LlaisysQwen2Model {
	LlaisysQwen2Meta meta{};
	LlaisysQwen2Weights weights{};
	llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
	std::vector<int> device_ids;
	std::unique_ptr<llaisys::models::Qwen2> impl;
	LlaisysQwen2KVContext *kv_ctx = nullptr; // experimental, non-decoder path
};

static void init_layer_arrays(LlaisysQwen2Weights &w, size_t nlayer) {
	w.attn_norm_w = new llaisysTensor_t[nlayer]();
	w.attn_q_w = new llaisysTensor_t[nlayer]();
	w.attn_q_b = new llaisysTensor_t[nlayer]();
	w.attn_k_w = new llaisysTensor_t[nlayer]();
	w.attn_k_b = new llaisysTensor_t[nlayer]();
	w.attn_v_w = new llaisysTensor_t[nlayer]();
	w.attn_v_b = new llaisysTensor_t[nlayer]();
	w.attn_o_w = new llaisysTensor_t[nlayer]();
	w.mlp_norm_w = new llaisysTensor_t[nlayer]();
	w.mlp_gate_w = new llaisysTensor_t[nlayer]();
	w.mlp_up_w = new llaisysTensor_t[nlayer]();
	w.mlp_down_w = new llaisysTensor_t[nlayer]();
}

static void destroy_layer_arrays(LlaisysQwen2Weights &w, size_t nlayer) {
	auto destroy_array = [nlayer](llaisysTensor_t *arr) {
		if (!arr) return;
		for (size_t i = 0; i < nlayer; ++i) {
			if (arr[i]) {
				tensorDestroy(arr[i]);
				arr[i] = nullptr;
			}
		}
		delete[] arr;
	};

	destroy_array(w.attn_norm_w);
	destroy_array(w.attn_q_w);
	destroy_array(w.attn_q_b);
	destroy_array(w.attn_k_w);
	destroy_array(w.attn_k_b);
	destroy_array(w.attn_v_w);
	destroy_array(w.attn_v_b);
	destroy_array(w.attn_o_w);
	destroy_array(w.mlp_norm_w);
	destroy_array(w.mlp_gate_w);
	destroy_array(w.mlp_up_w);
	destroy_array(w.mlp_down_w);

	w.attn_norm_w = nullptr;
	w.attn_q_w = nullptr;
	w.attn_q_b = nullptr;
	w.attn_k_w = nullptr;
	w.attn_k_b = nullptr;
	w.attn_v_w = nullptr;
	w.attn_v_b = nullptr;
	w.attn_o_w = nullptr;
	w.mlp_norm_w = nullptr;
	w.mlp_gate_w = nullptr;
	w.mlp_up_w = nullptr;
	w.mlp_down_w = nullptr;
}

__C {
	__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
		const LlaisysQwen2Meta *meta,
		llaisysDeviceType_t device,
		int *device_ids,
		int ndevice) {
		if (!meta || ndevice <= 0) return nullptr;

		auto *model = new LlaisysQwen2Model();
		model->meta = *meta;
		model->device = device;
		model->device_ids.assign(device_ids, device_ids + ndevice);

		init_layer_arrays(model->weights, model->meta.nlayer);
		model->impl = std::make_unique<llaisys::models::Qwen2>(
			model->meta,
			model->weights,
			model->device,
			model->device_ids);

		return model;
	}

    //销毁千问2模型实例
	__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
		if (!model) return;

		if (model->weights.in_embed) {
			tensorDestroy(model->weights.in_embed);
			model->weights.in_embed = nullptr;
		}
		if (model->weights.out_embed) {
			tensorDestroy(model->weights.out_embed);
			model->weights.out_embed = nullptr;
		}
		if (model->weights.out_norm_w) {
			tensorDestroy(model->weights.out_norm_w);
			model->weights.out_norm_w = nullptr;
		}

		destroy_layer_arrays(model->weights, model->meta.nlayer);
		if (model->kv_ctx) {
			llaisysQwen2KVContextRelease(model->kv_ctx);
			model->kv_ctx = nullptr;
		}

		model->impl.reset();
		delete model;
	}


    //获取千问2模型权重
	__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
		if (!model) return nullptr;
		return &model->weights;
	}

    //执行千问2模型推理
	__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
		if (!model || !model->impl) return -1;
		try {
			return model->impl->infer(token_ids, ntoken);
		} catch (const std::exception &e) {
			std::cerr << "[ERROR] Qwen2 infer failed: " << e.what() << std::endl;
			return -1;
		} catch (...) {
			std::cerr << "[ERROR] Qwen2 infer failed: unknown exception" << std::endl;
			return -1;
		}
	}

	__export int64_t llaisysQwen2ModelPrefill(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
		if (!model || !model->impl) return -1;
		try {
			return model->impl->prefill(token_ids, ntoken);
		} catch (const std::exception &e) {
			std::cerr << "[ERROR] Qwen2 prefill failed: " << e.what() << std::endl;
			return -1;
		} catch (...) {
			std::cerr << "[ERROR] Qwen2 prefill failed: unknown exception" << std::endl;
			return -1;
		}
	}

	__export int64_t llaisysQwen2ModelStep(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
		if (!model || !model->impl) return -1;
		try {
			return model->impl->step(token_ids, ntoken);
		} catch (const std::exception &e) {
			std::cerr << "[ERROR] Qwen2 step failed: " << e.what() << std::endl;
			return -1;
		} catch (...) {
			std::cerr << "[ERROR] Qwen2 step failed: unknown exception" << std::endl;
			return -1;
		}
	}

	__export int32_t llaisysQwen2ModelPrefillPacked(struct LlaisysQwen2Model *model,
	                                                int64_t *token_ids,
	                                                const int64_t *token_offsets,
	                                                size_t nseq,
	                                                int64_t *out_next_tokens) {
		if (!model || !model->impl || !token_ids || !token_offsets || !out_next_tokens || nseq == 0) return -1;
		try {
			const size_t ntoken = static_cast<size_t>(token_offsets[nseq]);
			if (!model->impl->prefillPacked(token_ids, ntoken, token_offsets, nseq, out_next_tokens)) return -2;
			return 0;
		} catch (const std::exception &e) {
			std::cerr << "[ERROR] Qwen2 prefill packed failed: " << e.what() << std::endl;
			return -3;
		} catch (...) {
			std::cerr << "[ERROR] Qwen2 prefill packed failed: unknown exception" << std::endl;
			return -4;
		}
	}

	__export int32_t llaisysQwen2ModelStepPacked(struct LlaisysQwen2Model *model,
	                                             int64_t *token_ids,
	                                             const int64_t *token_offsets,
	                                             size_t nseq,
	                                             int64_t *out_next_tokens) {
		if (!model || !model->impl || !token_ids || !token_offsets || !out_next_tokens || nseq == 0) return -1;
		try {
			const size_t ntoken = static_cast<size_t>(token_offsets[nseq]);
			if (!model->impl->stepPacked(token_ids, ntoken, token_offsets, nseq, out_next_tokens)) return -2;
			return 0;
		} catch (const std::exception &e) {
			std::cerr << "[ERROR] Qwen2 step packed failed: " << e.what() << std::endl;
			return -3;
		} catch (...) {
			std::cerr << "[ERROR] Qwen2 step packed failed: unknown exception" << std::endl;
			return -4;
		}
	}

	__export int64_t llaisysQwen2ModelPrefillSampling(struct LlaisysQwen2Model *model,
	                                                  int64_t *token_ids,
	                                                  size_t ntoken,
	                                                  const LlaisysSamplingParams *params) {
		if (!model || !model->impl) return -1;
		try {
			return model->impl->prefillSampling(token_ids, ntoken, params);
		} catch (const std::exception &e) {
			std::cerr << "[ERROR] Qwen2 prefill sampling failed: " << e.what() << std::endl;
			return -1;
		} catch (...) {
			std::cerr << "[ERROR] Qwen2 prefill sampling failed: unknown exception" << std::endl;
			return -1;
		}
	}

	__export int64_t llaisysQwen2ModelStepSampling(struct LlaisysQwen2Model *model,
	                                               int64_t *token_ids,
	                                               size_t ntoken,
	                                               const LlaisysSamplingParams *params) {
		if (!model || !model->impl) return -1;
		try {
			return model->impl->stepSampling(token_ids, ntoken, params);
		} catch (const std::exception &e) {
			std::cerr << "[ERROR] Qwen2 step sampling failed: " << e.what() << std::endl;
			return -1;
		} catch (...) {
			std::cerr << "[ERROR] Qwen2 step sampling failed: unknown exception" << std::endl;
			return -1;
		}
	}

	__export int64_t llaisysQwen2ModelInferSampling(struct LlaisysQwen2Model *model,
	                                                int64_t *token_ids,
	                                                size_t ntoken,
	                                                const LlaisysSamplingParams *params) {
		if (!model || !model->impl) return -1;
		try {
			return model->impl->prefillSampling(token_ids, ntoken, params);
		} catch (const std::exception &e) {
			std::cerr << "[ERROR] Qwen2 infer sampling failed: " << e.what() << std::endl;
			return -1;
		} catch (...) {
			std::cerr << "[ERROR] Qwen2 infer sampling failed: unknown exception" << std::endl;
			return -1;
		}
	}

	__export int64_t llaisysQwen2ModelInferSamplingEx(struct LlaisysQwen2Model *model,
	                                                  int64_t *token_ids,
	                                                  size_t ntoken,
	                                                  int32_t top_k,
	                                                  float top_p,
	                                                  float temperature,
	                                                  uint32_t seed) {
		if (!model || !model->impl) return -1;
		LlaisysSamplingParams params{};
		params.top_k = top_k;
		params.top_p = top_p;
		params.temperature = temperature;
		params.seed = seed;
		return llaisysQwen2ModelInferSampling(model, token_ids, ntoken, &params);
	}

	__export void llaisysQwen2ModelResetKVCache(struct LlaisysQwen2Model *model) {
		if (!model || !model->impl) return;
		model->impl->resetKVCache();
	}

	__export void llaisysQwen2ModelSetKVCacheEnabled(struct LlaisysQwen2Model *model, uint8_t enabled) {
		if (!model || !model->impl) return;
		model->impl->setKVCacheEnabled(enabled != 0);
	}

	__export int32_t llaisysQwen2ModelSetTensorParallel(struct LlaisysQwen2Model *model,
	                                                    llaisysComm_t comm,
	                                                    llaisysStream_t stream,
	                                                    int tp_size) {
		if (!model || !model->impl) return -1;
		model->impl->setTensorParallel(comm, stream, tp_size);
		return 0;
	}

	__export struct LlaisysQwen2KVBlock *llaisysQwen2KVBlockCreate(
		const struct LlaisysQwen2KVBlockMeta *meta,
		llaisysDeviceType_t device,
		int device_id) {
		if (!meta || meta->nlayer == 0 || meta->max_tokens == 0) return nullptr;
		auto *block = new LlaisysQwen2KVBlock();
		block->meta = *meta;
		block->device = device;
		block->device_id = device_id;
		block->k_layers.assign(meta->nlayer, nullptr);
		block->v_layers.assign(meta->nlayer, nullptr);
		size_t kv_shape[3] = {meta->max_tokens, meta->nkvh, meta->dh};
		for (size_t layer = 0; layer < meta->nlayer; ++layer) {
			block->k_layers[layer] = tensorCreate(kv_shape, 3, meta->dtype, device, device_id);
			block->v_layers[layer] = tensorCreate(kv_shape, 3, meta->dtype, device, device_id);
			if (!block->k_layers[layer] || !block->v_layers[layer]) {
				for (auto *t : block->k_layers) {
					if (t) tensorDestroy(t);
				}
				for (auto *t : block->v_layers) {
					if (t) tensorDestroy(t);
				}
				delete block;
				return nullptr;
			}
		}
		return block;
	}

	__export void llaisysQwen2KVBlockRetain(struct LlaisysQwen2KVBlock *block) {
		if (!block) return;
		block->ref_count.fetch_add(1, std::memory_order_relaxed);
	}

	__export void llaisysQwen2KVBlockRelease(struct LlaisysQwen2KVBlock *block) {
		if (!block) return;
		if (block->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			for (auto *t : block->k_layers) {
				if (t) tensorDestroy(t);
			}
			for (auto *t : block->v_layers) {
				if (t) tensorDestroy(t);
			}
			block->k_layers.clear();
			block->v_layers.clear();
			delete block;
		}
	}

	__export int32_t llaisysQwen2KVBlockSetTokenCount(struct LlaisysQwen2KVBlock *block, size_t used_tokens) {
		if (!block) return -1;
		if (used_tokens > block->meta.max_tokens) return -2;
		block->used_tokens = used_tokens;
		return 0;
	}

	__export size_t llaisysQwen2KVBlockTokenCount(const struct LlaisysQwen2KVBlock *block) {
		if (!block) return 0;
		return block->used_tokens;
	}

	__export llaisysTensor_t llaisysQwen2KVBlockKeyTensor(struct LlaisysQwen2KVBlock *block, size_t layer) {
		if (!block || layer >= block->k_layers.size()) return nullptr;
		return block->k_layers[layer];
	}

	__export llaisysTensor_t llaisysQwen2KVBlockValueTensor(struct LlaisysQwen2KVBlock *block, size_t layer) {
		if (!block || layer >= block->v_layers.size()) return nullptr;
		return block->v_layers[layer];
	}

	__export struct LlaisysQwen2KVContext *llaisysQwen2KVContextCreate(
		llaisysDataType_t dtype,
		llaisysDeviceType_t device,
		int device_id,
		size_t nlayer,
		size_t nh,
		size_t nkvh,
		size_t dh) {
		if (nlayer == 0 || dh == 0) return nullptr;
		auto *ctx = new LlaisysQwen2KVContext();
		ctx->dtype = dtype;
		ctx->device = device;
		ctx->device_id = device_id;
		ctx->nlayer = nlayer;
		ctx->nh = nh;
		ctx->nkvh = nkvh;
		ctx->dh = dh;
		return ctx;
	}

	__export void llaisysQwen2KVContextRetain(struct LlaisysQwen2KVContext *ctx) {
		if (!ctx) return;
		ctx->ref_count.fetch_add(1, std::memory_order_relaxed);
	}

	__export void llaisysQwen2KVContextRelease(struct LlaisysQwen2KVContext *ctx) {
		if (!ctx) return;
		if (ctx->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			for (auto *blk : ctx->chain) {
				llaisysQwen2KVBlockRelease(blk);
			}
			ctx->chain.clear();
			delete ctx;
		}
	}

	__export int32_t llaisysQwen2KVContextAttachBlock(
		struct LlaisysQwen2KVContext *ctx,
		struct LlaisysQwen2KVBlock *block) {
		if (!ctx || !block) return -1;
		if (ctx->device != block->device || ctx->device_id != block->device_id) return -2;
		if (ctx->dtype != block->meta.dtype) return -3;
		if (ctx->nlayer != block->meta.nlayer || ctx->dh != block->meta.dh) return -4;
		if (ctx->nkvh != block->meta.nkvh || ctx->nh != block->meta.nh) return -5;
		llaisysQwen2KVBlockRetain(block);
		ctx->chain.push_back(block);
		return 0;
	}

	__export void llaisysQwen2KVContextDetachAll(struct LlaisysQwen2KVContext *ctx) {
		if (!ctx) return;
		for (auto *blk : ctx->chain) {
			llaisysQwen2KVBlockRelease(blk);
		}
		ctx->chain.clear();
	}

	__export size_t llaisysQwen2KVContextBlockCount(const struct LlaisysQwen2KVContext *ctx) {
		if (!ctx) return 0;
		return ctx->chain.size();
	}

	__export size_t llaisysQwen2KVContextTokenCount(const struct LlaisysQwen2KVContext *ctx) {
		if (!ctx) return 0;
		size_t total = 0;
		for (auto *blk : ctx->chain) {
			if (!blk) continue;
			total += std::min(blk->used_tokens, blk->meta.max_tokens);
		}
		return total;
	}

	__export int32_t llaisysQwen2ModelSetKVContext(
		struct LlaisysQwen2Model *model,
		struct LlaisysQwen2KVContext *ctx) {
		if (!model) return -1;
		if (ctx) {
			if (model->device != ctx->device) return -2;
			const int model_device_id = model->device_ids.empty() ? 0 : model->device_ids[0];
			if (model_device_id != ctx->device_id) return -3;
			llaisysQwen2KVContextRetain(ctx);
		}
		if (model->kv_ctx) {
			llaisysQwen2KVContextRelease(model->kv_ctx);
		}
		model->kv_ctx = ctx;
		if (model->impl) {
			const size_t past_len_tokens = llaisysQwen2KVContextTokenCount(ctx);
			model->impl->setKVContext(ctx, past_len_tokens);
		}
		return 0;
	}

	__export struct LlaisysQwen2KVContext *llaisysQwen2ModelGetKVContext(
		struct LlaisysQwen2Model *model) {
		if (!model) return nullptr;
		auto *ctx = model->kv_ctx;
		if (model->impl) {
			ctx = reinterpret_cast<LlaisysQwen2KVContext *>(model->impl->getKVContext());
		}
		if (!ctx) return nullptr;
		llaisysQwen2KVContextRetain(ctx);
		return ctx;
	}

	__export int32_t llaisysQwen2ModelExportKVContext(
		struct LlaisysQwen2Model *model,
		struct LlaisysQwen2KVContext *ctx,
		size_t block_tokens) {
		if (!model || !model->impl || !ctx) return -1;
		if (model->device != ctx->device) return -2;
		const int model_device_id = model->device_ids.empty() ? 0 : model->device_ids[0];
		if (model_device_id != ctx->device_id) return -3;
		return static_cast<int32_t>(model->impl->exportKVContext(ctx, block_tokens));
	}
}
