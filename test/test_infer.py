from bootstrap import setup_paths

setup_paths(__file__)

import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys
import sys
import io
from llaisys.chat.service import build_chat_prompt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def resolve_model_path(model_path=None, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
        return model_path

    print(f"Loading model from Hugging Face: {model_id}")
    return snapshot_download(model_id)


def load_hf_model(
    model_path=None,
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device_name="cpu",
    strict_test=False,
):
    model_path = resolve_model_path(model_path, model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 严格一致性校验必须让 HF 与 LLAISYS 处在同一数值精度口径下，
    # 否则 GPU 上的 bf16 量化误差可能导致贪心解码 token 提前分叉。
    torch_dtype = torch.float32 if strict_test or device_name == "cpu" else torch.bfloat16
    device_map = None if device_name == "cpu" else torch_device(device_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device_name == "cpu":
        model = model.to(torch_device(device_name))

    return tokenizer, model, model_path


def build_input_content(prompt, tokenizer):
    return build_chat_prompt(
        tokenizer,
        [{"role": "user", "content": prompt}],
    )


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = build_input_content(prompt, tokenizer)
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


def load_llaisys_model(model_path, device_name):
    model = llaisys.models.load_model(model_path, llaisys_device(device_name))
    return model


def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = build_input_content(prompt, tokenizer)
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
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument(
        "--model_id",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        type=str,
    )
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer, model, model_path = load_hf_model(
        args.model,
        args.model_id,
        args.device,
        strict_test=args.test,
    )

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

    print("\n=== Answer ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    model = load_llaisys_model(model_path, args.device)
    start_time = time.time()
    llaisys_tokens, llaisys_output = llaisys_infer(
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
