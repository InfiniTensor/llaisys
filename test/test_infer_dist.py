import argparse
import gc
import os
import tempfile
import time
import uuid

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

import llaisys


def build_input_tokens(tokenizer, prompt: str):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(input_content)


def worker(rank: int, world_size: int, model_path: str, inputs, max_new_tokens: int, result_holder):
    dist = llaisys.DistributedContext()
    dist.init(rank, world_size)

    load_start = time.time()
    model = llaisys.models.Qwen2(
        model_path,
        llaisys.DeviceType.NVIDIA,
        device_id=rank,
        rank=rank,
        world_size=world_size,
    )
    load_elapsed = time.time() - load_start

    dist.barrier()
    generate_start = time.time()
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, top_k=1, top_p=1.0, temperature=1.0)
    dist.barrier()
    generate_elapsed = time.time() - generate_start

    if rank == 0:
        result_holder["tokens"] = outputs
        result_holder["load_latency"] = load_elapsed
        result_holder["generate_latency"] = generate_elapsed

    del model
    dist.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["nvidia"], type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=32, type=int)
    parser.add_argument("--world-size", default=8, type=int)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.device_count() >= args.world_size

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    inputs = build_input_tokens(tokenizer, args.prompt)

    single_load_start = time.time()
    single_model = llaisys.models.Qwen2(
        args.model,
        llaisys.DeviceType.NVIDIA,
        device_id=0,
        rank=0,
        world_size=1,
    )
    single_load_latency = time.time() - single_load_start
    start_time = time.time()
    single_tokens = single_model.generate(inputs, max_new_tokens=args.max_steps, top_k=1, top_p=1.0, temperature=1.0)
    single_generate_latency = time.time() - start_time
    del single_model
    gc.collect()
    torch.cuda.empty_cache()

    os.environ["LLAISYS_DIST_BOOTSTRAP_PATH"] = os.path.join(
        tempfile.gettempdir(), f"llaisys_qwen2_tp_{uuid.uuid4().hex}.bin"
    )

    manager = mp.Manager()
    result_holder = manager.dict()
    mp.spawn(worker, args=(args.world_size, args.model, inputs, args.max_steps, result_holder), nprocs=args.world_size, join=True)

    dist_tokens = list(result_holder["tokens"])
    dist_load_latency = float(result_holder["load_latency"])
    dist_generate_latency = float(result_holder["generate_latency"])

    print("Single-card tokens:")
    print(single_tokens)
    print(f"Single-card load latency: {single_load_latency:.2f}s")
    print(f"Single-card generate latency: {single_generate_latency:.2f}s")
    print("\nDistributed tokens:")
    print(dist_tokens)
    print(f"Distributed load latency: {dist_load_latency:.2f}s")
    print(f"Distributed generate latency: {dist_generate_latency:.2f}s")

    if args.test:
        assert dist_tokens == single_tokens
        print("\033[92mDistributed inference test passed!\033[0m\n")
