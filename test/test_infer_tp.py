import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import sys
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import llaisys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_hf_model(model_path=None, device_name="cuda"):
    # Check if local path exists
    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
        use_path = model_path
    else:
        # Try alternative local paths
        alt_paths = [
            "./models/deepseek-r1-distill-qwen-1.5b/",
            "../models/deepseek-r1-distill-qwen-1.5b/",
            "/home/hanson/llaisys/models/deepseek-r1-distill-qwen-1.5b/",
        ]
        use_path = None
        for p in alt_paths:
            if os.path.isdir(p):
                print(f"Loading model from local path: {p}")
                use_path = p
                model_path = p
                break
        
        if use_path is None:
            # Fall back to HuggingFace
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            print(f"Loading model from Hugging Face: {model_id}")
            use_path = model_id
            model_path = snapshot_download(model_id)
    
    tokenizer = AutoTokenizer.from_pretrained(use_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        use_path,
        dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result


def load_llaisys_tp_model(model_path, device_ids):
    print(f"Loading Tensor Parallel model on GPUs: {device_ids}")
    model = llaisys.models.Qwen2TP(model_path, device_ids=device_ids)
    return model


def llaisys_tp_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-ids", default="0,1", type=str,
                        help="Comma-separated GPU IDs for tensor parallelism")
    parser.add_argument("--model", default="./models/deepseek-r1-distill-qwen-1.5b/", type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    # Parse device IDs
    device_ids = [int(x.strip()) for x in args.device_ids.split(",")]
    if len(device_ids) < 2:
        print("Error: Tensor parallelism requires at least 2 GPUs")
        sys.exit(1)

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer, model, model_path = load_hf_model(args.model, "nvidia")

    # Example prompt
    start_time = time.time()
    tokens, output = hf_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    end_time = time.time()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== Answer ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    model = load_llaisys_tp_model(model_path, device_ids)
    start_time = time.time()
    llaisys_tokens, llaisys_output = llaisys_tp_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    end_time = time.time()

    print("\n=== Your Result ===\n")
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    if args.test:
        assert llaisys_tokens == tokens
        print("\033[92mTest passed!\033[0m\n")
