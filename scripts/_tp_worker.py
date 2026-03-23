#!/usr/bin/env python3
"""TP worker process -- spawned by launch_tp.py.

Reads env vars: RANK, WORLD_SIZE, CUDA_VISIBLE_DEVICES, TP_UID_FILE,
TP_MODEL_PATH, TP_DEVICE, TP_PROMPT, TP_MAX_TOKENS.
"""

import os
import sys
import ctypes
import time
from pathlib import Path

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)

import numpy as np
import safetensors

from llaisys.libllaisys import (
    LIB_LLAISYS,
    LlaisysCommAPI,
    llaisysComm_t,
    LLAISYS_COMM_UNIQUE_ID_MAX_SIZE,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
    llaisysDeviceType_t,
    llaisysDataType_t,
    LlaisysSamplingParams,
)
from llaisys.tensor_parallel import shard_qwen2_weights


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    uid_file = os.environ["TP_UID_FILE"]
    model_path = Path(os.environ["TP_MODEL_PATH"])
    device_name = os.environ.get("TP_DEVICE", "nvidia")
    prompt = os.environ.get("TP_PROMPT", "Hello")
    max_tokens = int(os.environ.get("TP_MAX_TOKENS", "64"))

    device = DeviceType.NVIDIA if device_name == "nvidia" else DeviceType.ILUVATAR
    backend = 0  # NCCL

    # Read unique ID
    for _ in range(100):
        if os.path.exists(uid_file) and os.path.getsize(uid_file) > 0:
            break
        time.sleep(0.1)
    with open(uid_file, "rb") as f:
        uid_bytes = f.read()

    # Init comm
    api_ptr = LIB_LLAISYS.llaisysGetCommAPI(backend)
    api = api_ptr.contents
    comm = llaisysComm_t()
    uid_buf = ctypes.create_string_buffer(uid_bytes, LLAISYS_COMM_UNIQUE_ID_MAX_SIZE)
    ret = api.init(ctypes.byref(comm), rank, world_size, uid_buf)
    if ret != 0:
        raise RuntimeError(f"commInit failed: {ret}")

    # Load tokenizer (use transformers for HF tokenizer.json)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    # Tokenize prompt
    import json
    config_path = model_path / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Load and shard weights
    weights = {}
    for file in sorted(model_path.glob("*.safetensors")):
        import torch
        data_ = safetensors.safe_open(file, framework="pt", device="cpu")
        for name in data_.keys():
            arr = data_.get_tensor(name)
            if arr.dtype == torch.bfloat16:
                arr = arr.to(torch.float16)
            weights[name] = arr.cpu().numpy()

    weights = shard_qwen2_weights(weights, rank, world_size)

    # Build model meta
    torch_dtype = str(cfg.get("torch_dtype", "bfloat16")).lower()
    dtype = DataType.F16  # we convert bf16->f16 above
    nlayer = int(cfg.get("num_hidden_layers", 0))
    hs = int(cfg.get("hidden_size", 0))
    nh = int(cfg.get("num_attention_heads", 0))
    nkvh = int(cfg.get("num_key_value_heads", nh))
    di = int(cfg.get("intermediate_size", 0))
    maxseq = int(cfg.get("max_position_embeddings", 0))
    voc = int(cfg.get("vocab_size", 0))
    epsilon = float(cfg.get("rms_norm_eps", 1e-6))
    theta = float(cfg.get("rope_theta", 10000.0))
    eos = cfg.get("eos_token_id", -1)
    end_token = int(eos[0]) if isinstance(eos, list) else int(eos)
    dh = int(cfg.get("head_dim", hs // nh if nh else 0))

    # Adjust nh/nkvh for TP
    tp_nh = nh // world_size
    tp_nkvh = nkvh // world_size

    model_meta = LlaisysQwen2Meta(
        llaisysDataType_t(dtype),
        ctypes.c_size_t(nlayer),
        ctypes.c_size_t(hs),
        ctypes.c_size_t(tp_nh),
        ctypes.c_size_t(tp_nkvh),
        ctypes.c_size_t(dh),
        ctypes.c_size_t(di // world_size),
        ctypes.c_size_t(maxseq),
        ctypes.c_size_t(voc),
        ctypes.c_float(epsilon),
        ctypes.c_float(theta),
        ctypes.c_int64(end_token),
    )

    device_ids = (ctypes.c_int * 1)(0)
    model = LIB_LLAISYS.llaisysQwen2ModelCreate(
        ctypes.byref(model_meta), llaisysDeviceType_t(device), device_ids, 1
    )
    if not model:
        raise RuntimeError("llaisysQwen2ModelCreate failed")

    LIB_LLAISYS.llaisysQwen2ModelSetKVCacheEnabled(model, ctypes.c_int(1))
    LIB_LLAISYS.llaisysQwen2ModelSetTensorParallel(model, comm, ctypes.c_void_p(None), world_size)
    model_weights = LIB_LLAISYS.llaisysQwen2ModelWeights(model)

    # Upload sharded weights
    def upload_tensor(arr):
        arr = np.ascontiguousarray(arr)
        shape = (ctypes.c_size_t * arr.ndim)(*arr.shape)
        dt = DataType.F16 if "float16" in arr.dtype.name else DataType.F32
        tensor = LIB_LLAISYS.tensorCreate(
            shape, ctypes.c_size_t(arr.ndim),
            llaisysDataType_t(dt), llaisysDeviceType_t(device), ctypes.c_int(0),
        )
        LIB_LLAISYS.tensorLoad(tensor, ctypes.c_void_p(arr.ctypes.data))
        return tensor

    w = model_weights.contents
    for name, arr in weights.items():
        tensor = upload_tensor(arr)
        if name in {"model.embed_tokens.weight", "transformer.wte.weight"}:
            w.in_embed = tensor
        elif name in {"lm_head.weight", "model.lm_head.weight"}:
            w.out_embed = tensor
        elif name in {"model.norm.weight", "transformer.ln_f.weight"}:
            w.out_norm_w = tensor
        elif name.startswith("model.layers."):
            parts = name.split(".")
            if len(parts) < 4:
                continue
            layer = int(parts[2])
            sub = ".".join(parts[3:])
            if sub == "input_layernorm.weight":
                w.attn_norm_w[layer] = tensor
            elif sub == "self_attn.q_proj.weight":
                w.attn_q_w[layer] = tensor
            elif sub == "self_attn.q_proj.bias":
                w.attn_q_b[layer] = tensor
            elif sub == "self_attn.k_proj.weight":
                w.attn_k_w[layer] = tensor
            elif sub == "self_attn.k_proj.bias":
                w.attn_k_b[layer] = tensor
            elif sub == "self_attn.v_proj.weight":
                w.attn_v_w[layer] = tensor
            elif sub == "self_attn.v_proj.bias":
                w.attn_v_b[layer] = tensor
            elif sub == "self_attn.o_proj.weight":
                w.attn_o_w[layer] = tensor
            elif sub == "post_attention_layernorm.weight":
                w.mlp_norm_w[layer] = tensor
            elif sub == "mlp.gate_proj.weight":
                w.mlp_gate_w[layer] = tensor
            elif sub == "mlp.up_proj.weight":
                w.mlp_up_w[layer] = tensor
            elif sub == "mlp.down_proj.weight":
                w.mlp_down_w[layer] = tensor

    if not w.out_embed and w.in_embed:
        w.out_embed = w.in_embed

    # Tokenize and run inference
    input_ids = tok.encode(prompt, add_special_tokens=True)

    # Prefill + decode
    token_buf = (ctypes.c_int64 * len(input_ids))(*input_ids)
    params = LlaisysSamplingParams(ctypes.c_int(1), ctypes.c_float(0.0), ctypes.c_float(0.0), ctypes.c_uint32(0))
    next_token = int(LIB_LLAISYS.llaisysQwen2ModelPrefillSampling(
        model, token_buf, ctypes.c_size_t(len(input_ids)), ctypes.byref(params),
    ))

    generated = list(input_ids)
    for step in range(max_tokens):
        if next_token < 0 or next_token == end_token:
            break
        generated.append(next_token)
        tb = (ctypes.c_int64 * 1)(next_token)
        next_token = int(LIB_LLAISYS.llaisysQwen2ModelStepSampling(
            model, tb, ctypes.c_size_t(1), ctypes.byref(params),
        ))

    # Decode and print from rank 0
    if rank == 0:
        output_text = tok.decode(generated, skip_special_tokens=True)
        print(output_text)

    # Cleanup
    LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
    api.destroy(comm)


if __name__ == "__main__":
    main()
