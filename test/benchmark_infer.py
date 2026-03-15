import argparse
import gc
import io
import logging
import os
import statistics
import sys
import time

import llaisys
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from test_utils import llaisys_device, torch_device

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROMPTS = {
    "short": "Who are you?",
    "medium": (
        "Explain the role of KV cache in transformer decoding, and give a short "
        "step-by-step example with one prompt token and two generated tokens."
    ),
    "long": (
        "I am building a tiny LLM inference system from scratch. Please provide a "
        "concise engineering checklist that covers model loading, tensor layout, "
        "runtime abstraction, memory reuse, operator profiling, and end-to-end "
        "benchmarking. Keep the answer practical and implementation-oriented."
    ),
}

logging.getLogger("transformers.dynamic_module_utils").setLevel(logging.ERROR)


def is_gpu_device(device_name):
    return device_name in {"nvidia", "metax"}


def parse_csv(text, caster=str):
    return [caster(x.strip()) for x in text.split(",") if x.strip()]


def load_hf_model(model_path, device_name):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    if model_path and os.path.isdir(model_path):
        model_path = os.path.expanduser(model_path)
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs = {"device_map": torch_device(device_name), "trust_remote_code": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, **kwargs)
    return tokenizer, model, model_path


def load_llaisys_model(model_path, device_name):
    return llaisys.models.Qwen2(model_path, llaisys_device(device_name))


def sync_torch(device_name):
    if is_gpu_device(device_name):
        torch.cuda.synchronize()


def sync_llaisys(device_name):
    llaisys.RuntimeAPI(llaisys_device(device_name)).device_synchronize()


def build_input_ids(tokenizer, prompt):
    text = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(text)


def run_torch_case(tokenizer, model, input_ids, max_new_tokens, top_k, top_p, temperature, device_name):
    inputs = torch.tensor(input_ids, dtype=torch.int64, device=model.device).unsqueeze(0)
    attention_mask = torch.ones_like(inputs)

    sync_torch(device_name)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    sync_torch(device_name)
    out_tokens = outputs[0].tolist()
    return time.perf_counter() - start, len(out_tokens) - len(input_ids), out_tokens


def run_llaisys_case(model, input_ids, max_new_tokens, top_k, top_p, temperature, device_name):
    sync_llaisys(device_name)
    start = time.perf_counter()
    out_tokens = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    sync_llaisys(device_name)
    return time.perf_counter() - start, len(out_tokens) - len(input_ids), out_tokens


def benchmark_backend(backend, tokenizer, model, cases, warmup, repeat, top_k, top_p, temperature, device_name):
    rows = {}
    for case in cases:
        for _ in range(warmup):
            if backend == "torch":
                run_torch_case(tokenizer, model, case["input_ids"], case["max_new_tokens"], top_k, top_p, temperature, device_name)
            else:
                run_llaisys_case(model, case["input_ids"], case["max_new_tokens"], top_k, top_p, temperature, device_name)

        latencies = []
        generated = []
        for _ in range(repeat):
            if backend == "torch":
                elapsed, new_tokens, _ = run_torch_case(
                    tokenizer, model, case["input_ids"], case["max_new_tokens"], top_k, top_p, temperature, device_name
                )
            else:
                elapsed, new_tokens, _ = run_llaisys_case(
                    model, case["input_ids"], case["max_new_tokens"], top_k, top_p, temperature, device_name
                )
            latencies.append(elapsed)
            generated.append(new_tokens)

        mean_s = statistics.mean(latencies)
        rows[(case["prompt_name"], case["max_new_tokens"])] = {
            "mean_ms": mean_s * 1000.0,
            "mean_new_tokens": statistics.mean(generated),
            "tokens_per_sec": statistics.mean(generated) / mean_s if mean_s > 0 else 0.0,
        }
    return rows


def print_report(cases, torch_rows, llaisys_rows):
    print("\n=== Torch vs LLAISYS Inference Benchmark ===")
    print("| Case | Torch mean(ms) | Torch tok/s | LLAISYS mean(ms) | LLAISYS tok/s | speedup |")
    print("|---|---:|---:|---:|---:|---:|")

    torch_total_tokens = 0.0
    llaisys_total_tokens = 0.0
    torch_total_seconds = 0.0
    llaisys_total_seconds = 0.0

    for case in cases:
        key = (case["prompt_name"], case["max_new_tokens"])
        torch_row = torch_rows[key]
        llaisys_row = llaisys_rows[key]
        speedup = torch_row["mean_ms"] / llaisys_row["mean_ms"] if llaisys_row["mean_ms"] > 0 else 0.0

        print(
            f"| {case['prompt_name']}/{case['max_new_tokens']} | {torch_row['mean_ms']:.2f} | {torch_row['tokens_per_sec']:.2f} | "
            f"{llaisys_row['mean_ms']:.2f} | {llaisys_row['tokens_per_sec']:.2f} | {speedup:.2f}x |"
        )

        torch_total_tokens += torch_row["mean_new_tokens"]
        llaisys_total_tokens += llaisys_row["mean_new_tokens"]
        torch_total_seconds += torch_row["mean_ms"] / 1000.0
        llaisys_total_seconds += llaisys_row["mean_ms"] / 1000.0

    torch_total_tok_s = torch_total_tokens / torch_total_seconds if torch_total_seconds > 0 else 0.0
    llaisys_total_tok_s = llaisys_total_tokens / llaisys_total_seconds if llaisys_total_seconds > 0 else 0.0
    overall_speedup = llaisys_total_tok_s / torch_total_tok_s if torch_total_tok_s > 0 else 0.0

    print("\n=== Throughput Summary ===")
    print(f"Torch total throughput   : {torch_total_tok_s:.2f} tok/s")
    print(f"LLAISYS total throughput : {llaisys_total_tok_s:.2f} tok/s")
    print(f"Overall speedup          : {overall_speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Torch vs LLAISYS inference throughput.")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia", "metax"], type=str)
    parser.add_argument("--prompts", default="short,medium,long", type=str)
    parser.add_argument("--max-new-tokens", default="32,64,128", type=str)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--top-k", default=1, type=int)
    parser.add_argument("--top-p", default=1.0, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    args = parser.parse_args()

    top_k, top_p, temperature = args.top_k, args.top_p, args.temperature

    prompt_names = parse_csv(args.prompts)
    max_new_tokens_list = parse_csv(args.max_new_tokens, int)
    for name in prompt_names:
        if name not in PROMPTS:
            raise ValueError(f"Unknown prompt preset: {name}. Valid keys: {list(PROMPTS.keys())}")

    tokenizer, torch_model, model_path = load_hf_model(args.model, args.device)
    cases = [
        {
            "prompt_name": prompt_name,
            "max_new_tokens": max_new_tokens,
            "input_ids": build_input_ids(tokenizer, PROMPTS[prompt_name]),
        }
        for prompt_name in prompt_names
        for max_new_tokens in max_new_tokens_list
    ]

    torch_rows = benchmark_backend(
        "torch", tokenizer, torch_model, cases, args.warmup, args.repeat, top_k, top_p, temperature, args.device
    )

    del torch_model
    gc.collect()
    if is_gpu_device(args.device):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    llaisys_model = load_llaisys_model(model_path, args.device)
    llaisys_rows = benchmark_backend(
        "llaisys", tokenizer, llaisys_model, cases, args.warmup, args.repeat, top_k, top_p, temperature, args.device
    )

    print_report(cases, torch_rows, llaisys_rows)


if __name__ == "__main__":
    main()
