#!/usr/bin/env python
import argparse
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import torch

import llaisys
from llaisys.chat.service import build_chat_prompt
from llaisys.models import load_model
from transformers import AutoTokenizer


def torch_device(device_name: str):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name in ("nvidia", "metax"):
        return torch.device("cuda:0")
    raise ValueError(f"Unsupported device: {device_name}")


def llaisys_device(device_name: str):
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    if device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    if device_name == "metax":
        return llaisys.DeviceType.METAX
    raise ValueError(f"Unsupported device: {device_name}")


def sync_device(device_name: str):
    llaisys.RuntimeAPI(llaisys_device(device_name)).device_synchronize()
    if device_name in ("nvidia", "metax"):
        torch.cuda.synchronize()


def benchmark_linear(device_name: str, repeat: int):
    x = torch.rand((512, 4096), dtype=torch.float32, device=torch_device(device_name)) * 0.1
    w = torch.rand((4096, 4096), dtype=torch.float32, device=torch_device(device_name)) * 0.01
    bias = torch.rand((4096,), dtype=torch.float32, device=torch_device(device_name))
    out = torch.empty((512, 4096), dtype=torch.float32, device=torch_device(device_name))

    x_ll = llaisys.Tensor((512, 4096), dtype=llaisys.DataType.F32, device=llaisys_device(device_name))
    w_ll = llaisys.Tensor((4096, 4096), dtype=llaisys.DataType.F32, device=llaisys_device(device_name))
    b_ll = llaisys.Tensor((4096,), dtype=llaisys.DataType.F32, device=llaisys_device(device_name))
    o_ll = llaisys.Tensor((512, 4096), dtype=llaisys.DataType.F32, device=llaisys_device(device_name))
    runtime = llaisys.RuntimeAPI(llaisys_device(device_name))
    runtime.memcpy_sync(x_ll.data_ptr(), x.data_ptr(), x.numel() * x.element_size(), llaisys.MemcpyKind.D2D)
    runtime.memcpy_sync(w_ll.data_ptr(), w.data_ptr(), w.numel() * w.element_size(), llaisys.MemcpyKind.D2D)
    runtime.memcpy_sync(b_ll.data_ptr(), bias.data_ptr(), bias.numel() * bias.element_size(), llaisys.MemcpyKind.D2D)

    # 预热一轮，避免首次调用的 lazy init 影响测量。
    torch.nn.functional.linear(x, w, bias, out=out)
    llaisys.Ops.linear(o_ll, x_ll, w_ll, b_ll)
    sync_device(device_name)

    start = time.time()
    for _ in range(repeat):
        torch.nn.functional.linear(x, w, bias, out=out)
    sync_device(device_name)
    torch_ms = (time.time() - start) * 1000.0 / repeat

    start = time.time()
    for _ in range(repeat):
        llaisys.Ops.linear(o_ll, x_ll, w_ll, b_ll)
    sync_device(device_name)
    llaisys_ms = (time.time() - start) * 1000.0 / repeat

    print("=== Linear Benchmark ===")
    print(f"device: {device_name}")
    print(f"shape: x=(512,4096), w=(4096,4096), out=(512,4096)")
    print(f"torch_avg_ms: {torch_ms:.3f}")
    print(f"llaisys_avg_ms: {llaisys_ms:.3f}")


def benchmark_infer(device_name: str, model_path: str, prompt: str, max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path, llaisys_device(device_name))
    prompt_text = build_chat_prompt(tokenizer, [{"role": "user", "content": prompt}])
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    start = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    sync_device(device_name)
    elapsed = time.time() - start
    completion_ids = output_ids[len(input_ids):]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    print("\n=== Inference Benchmark ===")
    print(f"device: {device_name}")
    print(f"model_path: {model_path}")
    print(f"prompt_tokens: {len(input_ids)}")
    print(f"completion_tokens: {len(completion_ids)}")
    print(f"elapsed_s: {elapsed:.3f}")
    print("completion_preview:")
    print(completion_text[:400])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"])
    parser.add_argument("--repeat", default=20, type=int)
    parser.add_argument("--model", default="", help="可选，本地模型目录")
    parser.add_argument("--prompt", default="请用中文介绍一下你自己。")
    parser.add_argument("--max-new-tokens", default=64, type=int)
    args = parser.parse_args()

    if args.device in ("nvidia", "metax"):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    benchmark_linear(args.device, args.repeat)
    if args.model:
        benchmark_infer(args.device, args.model, args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
