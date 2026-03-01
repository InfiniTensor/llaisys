#include "qwen2_tp.hpp"
#include <cstring>
#include <cmath>
#include <iostream>

namespace llaisys {
namespace models {

// Forward declare CUDA runtime functions
typedef int cudaError_t;
#define cudaSuccess 0
extern "C" cudaError_t cudaSetDevice(int device);
extern "C" cudaError_t cudaDeviceSynchronize(void);
extern "C" cudaError_t cudaGetLastError(void);
extern "C" cudaError_t cudaGetDeviceCount(int* count);
extern "C" cudaError_t cudaFree(void* devPtr);
extern "C" const char* cudaGetErrorString(cudaError_t error);

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
    } \
} while(0)

// ==================== Qwen2ModelShard ====================

Qwen2ModelShard::Qwen2ModelShard(int rank_, int device_id_)
    : rank(rank_), device_id(device_id_), communicator(nullptr) {
}

void Qwen2ModelShard::allocateWeights(const LlaisysQwen2Meta& meta, int world_size) {
    auto dtype = meta.dtype;
    size_t nlayer = meta.nlayer;
    size_t hs = meta.hs;
    size_t di = meta.di;
    size_t voc = meta.voc;
    size_t dh = meta.dh;
    size_t nh = meta.nh;
    size_t nkvh = meta.nkvh;

    size_t nh_shard = nh / world_size;
    size_t nkvh_shard = nkvh / world_size;
    size_t di_shard = di / world_size;

    in_embed = Tensor::create({voc, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    out_embed = Tensor::create({voc, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    out_norm_w = Tensor::create({hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);

    attn_norm_w.resize(nlayer);
    attn_q_w.resize(nlayer);
    attn_q_b.resize(nlayer);
    attn_k_w.resize(nlayer);
    attn_k_b.resize(nlayer);
    attn_v_w.resize(nlayer);
    attn_v_b.resize(nlayer);
    attn_o_w.resize(nlayer);
    mlp_norm_w.resize(nlayer);
    mlp_gate_w.resize(nlayer);
    mlp_up_w.resize(nlayer);
    mlp_down_w.resize(nlayer);

    for (size_t i = 0; i < nlayer; ++i) {
        attn_norm_w[i] = Tensor::create({hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
        attn_q_w[i] = Tensor::create({nh_shard * dh, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
        attn_q_b[i] = Tensor::create({nh_shard * dh}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
        attn_k_w[i] = Tensor::create({nkvh_shard * dh, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
        attn_k_b[i] = Tensor::create({nkvh_shard * dh}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
        attn_v_w[i] = Tensor::create({nkvh_shard * dh, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
        attn_v_b[i] = Tensor::create({nkvh_shard * dh}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
        attn_o_w[i] = Tensor::create({hs, nh_shard * dh}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
        mlp_norm_w[i] = Tensor::create({hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
        mlp_gate_w[i] = Tensor::create({di_shard, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
        mlp_up_w[i] = Tensor::create({di_shard, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
        mlp_down_w[i] = Tensor::create({hs, di_shard}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    }
}

void Qwen2ModelShard::allocateCache(const LlaisysQwen2Meta& meta, int world_size) {
    size_t nlayer = meta.nlayer;
    size_t maxseq = meta.maxseq;
    size_t nkvh = meta.nkvh;
    size_t dh = meta.dh;
    size_t nkvh_shard = nkvh / world_size;
    auto dtype_val = meta.dtype;

    k_cache.resize(nlayer);
    v_cache.resize(nlayer);

    for (size_t i = 0; i < nlayer; ++i) {
        k_cache[i] = Tensor::create({maxseq, nkvh_shard, dh}, dtype_val, LLAISYS_DEVICE_NVIDIA, device_id);
        v_cache[i] = Tensor::create({maxseq, nkvh_shard, dh}, dtype_val, LLAISYS_DEVICE_NVIDIA, device_id);
    }
}

void Qwen2ModelShard::allocateBuffers(const LlaisysQwen2Meta& meta, size_t max_seqlen, int world_size) {
    auto dtype = meta.dtype;
    size_t hs = meta.hs;
    size_t dh = meta.dh;
    size_t di = meta.di;
    size_t voc = meta.voc;
    size_t nh = meta.nh;
    size_t nkvh = meta.nkvh;
    size_t nh_shard = nh / world_size;
    size_t nkvh_shard = nkvh / world_size;
    size_t di_shard = di / world_size;

    hidden = Tensor::create({max_seqlen, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    hidden_norm = Tensor::create({max_seqlen, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    q = Tensor::create({max_seqlen, nh_shard * dh}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    k = Tensor::create({max_seqlen, nkvh_shard * dh}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    v = Tensor::create({max_seqlen, nkvh_shard * dh}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    q_rope = Tensor::create({max_seqlen, nh_shard, dh}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    k_rope = Tensor::create({max_seqlen, nkvh_shard, dh}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    attn_out = Tensor::create({max_seqlen, nh_shard, dh}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    attn_proj = Tensor::create({max_seqlen, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    gate = Tensor::create({max_seqlen, di_shard}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    up = Tensor::create({max_seqlen, di_shard}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    mlp_out = Tensor::create({max_seqlen, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    logits = Tensor::create({1, voc}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_NVIDIA, device_id);
    max_val = Tensor::create({1}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
    pos_ids = Tensor::create({max_seqlen}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_NVIDIA, device_id);
    allreduce_buffer = Tensor::create({max_seqlen, hs}, dtype, LLAISYS_DEVICE_NVIDIA, device_id);
}

// ==================== Qwen2ModelTP ====================

Qwen2ModelTP::Qwen2ModelTP(const LlaisysQwen2Meta* meta_, const std::vector<int>& device_ids_)
    : world_size(static_cast<int>(device_ids_.size())),
      device_ids(device_ids_), 
      cache_len(0),
      workers_running(false),
      task_queues(device_ids_.size()),
      queue_mutexes(device_ids_.size()),
      queue_cvs(device_ids_.size()),
      infer_barrier(nullptr),
      infer_result(0) {
    
    std::memcpy(&meta, meta_, sizeof(LlaisysQwen2Meta));

    if (meta.nh % world_size != 0) {
        throw std::invalid_argument("nh must be divisible by world_size");
    }
    if (meta.nkvh % world_size != 0) {
        throw std::invalid_argument("nkvh must be divisible by world_size");
    }
    if (meta.di % world_size != 0) {
        throw std::invalid_argument("di must be divisible by world_size");
    }

    initialize();
}

Qwen2ModelTP::~Qwen2ModelTP() {
    stopWorkerThreads();
}

void Qwen2ModelTP::initialize() {
    // Initialize CUDA driver and get device count
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    // Initialize CUDA context on first device
    CUDA_CHECK(cudaSetDevice(device_ids[0]));
    cudaFree(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    core::context().setDevice(LLAISYS_DEVICE_NVIDIA, device_ids[0]);
    
    // Create NCCL communicators
    communicators = device::nvidia::NCCLCommunicator::createAll(device_ids);
    
    // Reset to first device after NCCL init
    CUDA_CHECK(cudaSetDevice(device_ids[0]));
    core::context().setDevice(LLAISYS_DEVICE_NVIDIA, device_ids[0]);

    // Create shards with proper device context
    shards.reserve(world_size);
    for (int i = 0; i < world_size; ++i) {
        CUDA_CHECK(cudaSetDevice(device_ids[i]));
        core::context().setDevice(LLAISYS_DEVICE_NVIDIA, device_ids[i]);
        
        shards.push_back(std::make_unique<Qwen2ModelShard>(i, device_ids[i]));
        shards[i]->communicator = communicators[i].get();
        shards[i]->allocateWeights(meta, world_size);
        shards[i]->allocateCache(meta, world_size);
        shards[i]->allocateBuffers(meta, meta.maxseq, world_size);
    }
    
    // Reset to first device after initialization
    CUDA_CHECK(cudaSetDevice(device_ids[0]));
    core::context().setDevice(LLAISYS_DEVICE_NVIDIA, device_ids[0]);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Start persistent worker threads
    startWorkerThreads();
    
    // Run 1-token warm-up to initialize NCCL internal state
    // This prevents "unhandled cuda error" on first real inference
    alignas(64) int64_t warmup_token[1] = {1};
    infer(warmup_token, 1);
    cache_len.store(0);  // Reset cache after warm-up
}

void Qwen2ModelTP::startWorkerThreads() {
    workers_running.store(true);
    infer_barrier = new SimpleBarrier(world_size);
    
    worker_threads.reserve(world_size);
    for (int rank = 0; rank < world_size; ++rank) {
        worker_threads.emplace_back(&Qwen2ModelTP::workerLoop, this, rank);
    }
    
    // Wait for all workers to initialize their CUDA context
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Synchronize all devices
    for (int i = 0; i < world_size; ++i) {
        CUDA_CHECK(cudaSetDevice(device_ids[i]));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaSetDevice(device_ids[0]));
}

void Qwen2ModelTP::stopWorkerThreads() {
    if (!workers_running.load()) return;
    
    workers_running.store(false);
    
    // Wake up all worker threads
    for (int i = 0; i < world_size; ++i) {
        std::unique_lock<std::mutex> lock(queue_mutexes[i]);
        queue_cvs[i].notify_all();
    }
    
    // Join all threads
    for (auto& t : worker_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    worker_threads.clear();
    delete infer_barrier;
    infer_barrier = nullptr;
}

void Qwen2ModelTP::workerLoop(int rank) {
    // CRITICAL: Set CUDA device and context for this thread
    // This ensures the thread's CUDA context matches the communicator's context
    CUDA_CHECK(cudaSetDevice(device_ids[rank]));
    CUDA_CHECK(cudaDeviceSynchronize());
    core::context().setDevice(LLAISYS_DEVICE_NVIDIA, device_ids[rank]);
    
    auto& shard = *shards[rank];
    
    // Synchronize to ensure all workers have set their device
    CUDA_CHECK(cudaDeviceSynchronize());
    
    while (workers_running.load()) {
        // Wait for task
        std::unique_lock<std::mutex> lock(queue_mutexes[rank]);
        queue_cvs[rank].wait(lock, [this, rank] { 
            return !task_queues[rank].empty() || !workers_running.load(); 
        });
        
        if (!workers_running.load()) break;
        
        if (task_queues[rank].empty()) continue;
        
        // Get task
        InferTask task = std::move(task_queues[rank].front());
        task_queues[rank].pop();
        lock.unlock();
        
        // Execute inference
        size_t ntoken = task.ntoken;
        size_t start_pos = task.cache_len;
        
        // Ensure device context is set before computation
        CUDA_CHECK(cudaSetDevice(device_ids[rank]));
        core::context().setDevice(LLAISYS_DEVICE_NVIDIA, device_ids[rank]);
        
        // Create input tensor and run embedding
        tensor_t input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, 
                                            LLAISYS_DEVICE_NVIDIA, shard.device_id);
        
        alignas(64) int64_t aligned_tokens[4096];
        if (ntoken <= 4096) {
            std::memcpy(aligned_tokens, task.token_ids, ntoken * sizeof(int64_t));
            input_ids->load(aligned_tokens);
        } else {
            input_ids->load(task.token_ids);
        }
        
        tensor_t hidden_view = shard.hidden->slice(0, 0, ntoken);
        ops::embedding(hidden_view, input_ids, shard.in_embed);
        
        // Forward through all layers
        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            forwardLayerWithBarrier(rank, layer, ntoken, start_pos, *infer_barrier);
        }
        
        // Only rank 0 computes the final output
        if (rank == 0) {
            cache_len += ntoken;
            
            tensor_t last_hidden = shard.hidden->slice(0, ntoken - 1, ntoken);
            tensor_t last_norm = shard.hidden_norm->slice(0, 0, 1);
            
            ops::rms_norm(last_norm, last_hidden, shard.out_norm_w, meta.epsilon);
            ops::linear(shard.logits, last_norm, shard.out_embed, nullptr);
            
            tensor_t last_logits = shard.logits->view({meta.voc});
            ops::argmax(shard.max_idx, shard.max_val, last_logits);
            
            alignas(64) int64_t result;
            core::context().runtime().api()->memcpy_sync(
                &result, shard.max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
            
            infer_result.store(result);
            
            if (task.result_promise) {
                task.result_promise->set_value(result);
            }
        }
        
        // Synchronize all ranks at the end
        infer_barrier->arrive_and_wait();
    }
}

void Qwen2ModelTP::forwardLayerWithBarrier(int rank, size_t layer, size_t seqlen, size_t start_pos, SimpleBarrier& barrier) {
    auto& shard = *shards[rank];
    size_t total_len = start_pos + seqlen;
    size_t nh_shard = meta.nh / world_size;
    size_t nkvh_shard = meta.nkvh / world_size;
    size_t dh = meta.dh;

    tensor_t hidden_view = shard.hidden->slice(0, 0, seqlen);
    tensor_t norm_view = shard.hidden_norm->slice(0, 0, seqlen);
    tensor_t q_view = shard.q->slice(0, 0, seqlen);
    tensor_t k_view = shard.k->slice(0, 0, seqlen);
    tensor_t v_view = shard.v->slice(0, 0, seqlen);
    tensor_t q_rope_view = shard.q_rope->slice(0, 0, seqlen);
    tensor_t k_rope_view = shard.k_rope->slice(0, 0, seqlen);
    tensor_t attn_out_view = shard.attn_out->slice(0, 0, seqlen);
    tensor_t attn_proj_view = shard.attn_proj->slice(0, 0, seqlen);
    tensor_t gate_view = shard.gate->slice(0, 0, seqlen);
    tensor_t up_view = shard.up->slice(0, 0, seqlen);

    ops::rms_norm(norm_view, hidden_view, shard.attn_norm_w[layer], meta.epsilon);
    ops::linear(q_view, norm_view, shard.attn_q_w[layer], shard.attn_q_b[layer]);
    ops::linear(k_view, norm_view, shard.attn_k_w[layer], shard.attn_k_b[layer]);
    ops::linear(v_view, norm_view, shard.attn_v_w[layer], shard.attn_v_b[layer]);

    tensor_t q_reshaped = q_view->view({seqlen, nh_shard, dh});
    tensor_t k_reshaped = k_view->view({seqlen, nkvh_shard, dh});
    tensor_t v_reshaped = v_view->view({seqlen, nkvh_shard, dh});

    tensor_t pos_view = shard.pos_ids->slice(0, 0, seqlen);
    alignas(64) int64_t pos_data[4096];
    for (size_t i = 0; i < seqlen; ++i) {
        pos_data[i] = static_cast<int64_t>(start_pos + i);
    }
    pos_view->load(pos_data);

    ops::rope(q_rope_view, q_reshaped, pos_view, meta.theta);
    ops::rope(k_rope_view, k_reshaped, pos_view, meta.theta);

    tensor_t k_cache_slice = shard.k_cache[layer]->slice(0, start_pos, total_len);
    tensor_t v_cache_slice = shard.v_cache[layer]->slice(0, start_pos, total_len);

    size_t kv_bytes = seqlen * nkvh_shard * dh * k_rope_view->elementSize();
    auto api = core::context().runtime().api();
    api->memcpy_sync(k_cache_slice->data(), k_rope_view->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
    api->memcpy_sync(v_cache_slice->data(), v_reshaped->data(), kv_bytes, LLAISYS_MEMCPY_D2D);

    tensor_t k_full = shard.k_cache[layer]->slice(0, 0, total_len);
    tensor_t v_full = shard.v_cache[layer]->slice(0, 0, total_len);

    float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    ops::self_attention(attn_out_view, q_rope_view, k_full, v_full, scale);

    tensor_t attn_out_flat = attn_out_view->view({seqlen, nh_shard * dh});
    ops::linear(attn_proj_view, attn_out_flat, shard.attn_o_w[layer], nullptr);

    size_t hidden_bytes = seqlen * meta.hs * hidden_view->elementSize();
    api->memcpy_sync(shard.allreduce_buffer->data(), attn_proj_view->data(), 
                     hidden_bytes, LLAISYS_MEMCPY_D2D);
    
    barrier.arrive_and_wait();
    
    // CRITICAL: Set device context before NCCL operation
    CUDA_CHECK(cudaSetDevice(shard.device_id));
    core::context().setDevice(LLAISYS_DEVICE_NVIDIA, shard.device_id);
    shard.communicator->allReduce(shard.allreduce_buffer->data(), seqlen * meta.hs, meta.dtype);
    barrier.arrive_and_wait();
    
    api->memcpy_sync(attn_proj_view->data(), shard.allreduce_buffer->data(),
                     hidden_bytes, LLAISYS_MEMCPY_D2D);

    ops::add(hidden_view, hidden_view, attn_proj_view);
    ops::rms_norm(norm_view, hidden_view, shard.mlp_norm_w[layer], meta.epsilon);
    ops::linear(gate_view, norm_view, shard.mlp_gate_w[layer], nullptr);
    ops::linear(up_view, norm_view, shard.mlp_up_w[layer], nullptr);
    ops::swiglu(gate_view, gate_view, up_view);
    ops::linear(attn_proj_view, gate_view, shard.mlp_down_w[layer], nullptr);

    api->memcpy_sync(shard.allreduce_buffer->data(), attn_proj_view->data(),
                     hidden_bytes, LLAISYS_MEMCPY_D2D);
    
    barrier.arrive_and_wait();
    
    // CRITICAL: Set device context before NCCL operation
    CUDA_CHECK(cudaSetDevice(shard.device_id));
    core::context().setDevice(LLAISYS_DEVICE_NVIDIA, shard.device_id);
    shard.communicator->allReduce(shard.allreduce_buffer->data(), seqlen * meta.hs, meta.dtype);
    barrier.arrive_and_wait();

    api->memcpy_sync(attn_proj_view->data(), shard.allreduce_buffer->data(),
                     hidden_bytes, LLAISYS_MEMCPY_D2D);

    ops::add(hidden_view, hidden_view, attn_proj_view);
}

void Qwen2ModelTP::allReduce(int rank, void* buffer, size_t count, int dtype) {
    CUDA_CHECK(cudaSetDevice(shards[rank]->device_id));
    CUDA_CHECK(cudaDeviceSynchronize());
    communicators[rank]->allReduce(buffer, count, dtype);
}

int64_t Qwen2ModelTP::infer(int64_t* token_ids, size_t ntoken) {
    std::promise<int64_t> result_promise;
    std::future<int64_t> result_future = result_promise.get_future();
    
    // Create tasks for all ranks
    for (int rank = 0; rank < world_size; ++rank) {
        std::unique_lock<std::mutex> lock(queue_mutexes[rank]);
        
        InferTask task;
        task.token_ids = token_ids;
        task.ntoken = ntoken;
        task.cache_len = cache_len.load();
        // Only rank 0 gets the promise
        task.result_promise = (rank == 0) ? &result_promise : nullptr;
        
        task_queues[rank].push(std::move(task));
        lock.unlock();
        queue_cvs[rank].notify_one();
    }
    
    // Wait for result from rank 0
    return result_future.get();
}

void Qwen2ModelTP::resetCache() {
    cache_len.store(0);
}

LlaisysQwen2Weights* Qwen2ModelTP::getWeights(int rank) {
    auto* weights = new LlaisysQwen2Weights();
    auto& shard = *shards[rank];

    weights->in_embed = new LlaisysTensor{shard.in_embed};
    weights->out_embed = new LlaisysTensor{shard.out_embed};
    weights->out_norm_w = new LlaisysTensor{shard.out_norm_w};

    // Allocate C-style arrays for per-layer weights
    weights->attn_norm_w = new llaisysTensor_t[meta.nlayer];
    weights->attn_q_w = new llaisysTensor_t[meta.nlayer];
    weights->attn_q_b = new llaisysTensor_t[meta.nlayer];
    weights->attn_k_w = new llaisysTensor_t[meta.nlayer];
    weights->attn_k_b = new llaisysTensor_t[meta.nlayer];
    weights->attn_v_w = new llaisysTensor_t[meta.nlayer];
    weights->attn_v_b = new llaisysTensor_t[meta.nlayer];
    weights->attn_o_w = new llaisysTensor_t[meta.nlayer];
    weights->mlp_norm_w = new llaisysTensor_t[meta.nlayer];
    weights->mlp_gate_w = new llaisysTensor_t[meta.nlayer];
    weights->mlp_up_w = new llaisysTensor_t[meta.nlayer];
    weights->mlp_down_w = new llaisysTensor_t[meta.nlayer];

    for (size_t i = 0; i < meta.nlayer; ++i) {
        weights->attn_norm_w[i] = new LlaisysTensor{shard.attn_norm_w[i]};
        weights->attn_q_w[i] = new LlaisysTensor{shard.attn_q_w[i]};
        weights->attn_q_b[i] = new LlaisysTensor{shard.attn_q_b[i]};
        weights->attn_k_w[i] = new LlaisysTensor{shard.attn_k_w[i]};
        weights->attn_k_b[i] = new LlaisysTensor{shard.attn_k_b[i]};
        weights->attn_v_w[i] = new LlaisysTensor{shard.attn_v_w[i]};
        weights->attn_v_b[i] = new LlaisysTensor{shard.attn_v_b[i]};
        weights->attn_o_w[i] = new LlaisysTensor{shard.attn_o_w[i]};
        weights->mlp_norm_w[i] = new LlaisysTensor{shard.mlp_norm_w[i]};
        weights->mlp_gate_w[i] = new LlaisysTensor{shard.mlp_gate_w[i]};
        weights->mlp_up_w[i] = new LlaisysTensor{shard.mlp_up_w[i]};
        weights->mlp_down_w[i] = new LlaisysTensor{shard.mlp_down_w[i]};
    }

    return weights;
}

} // namespace models
} // namespace llaisys
