#!/usr/bin/env python3
"""
GPU 逐层诊断：找出首个产生错误结果的 GPU 层。
要求：已 xmake && xmake install，且编译时启用 NVIDIA（xmake f --nv-gpu=y）。
用法：
  cd /home/chenncy/llaisys
  LLAISYS_GPU_FULL_CPU=1 .venv/bin/python test/diagnose_gpu_layer.py --model /home/chenncy/llaisys/DeepSeek-R1-Distill-Qwen-1___5B
  或
  LLAISYS_GPU_FULL_CPU=1 PYTHONPATH=./python python test/diagnose_gpu_layer.py --model <模型目录>
"""
import os
import sys
import argparse

# 必须在 import llaisys_py 之前设置，这样 Qwen2 加载时会调用 CacheAllWeightsOnCPU
os.environ["LLAISYS_GPU_FULL_CPU"] = "1"

from llaisys_py.models.qwen2 import Qwen2
from llaisys_py.libllaisys import DeviceType


def main():
    parser = argparse.ArgumentParser(description="GPU 逐层诊断：找出首个出错的 GPU 层")
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL_PATH", ""), help="模型目录（含 config.json 与 safetensors）")
    parser.add_argument("--prompt", type=str, default="什么是", help="用于 prefill 的短句（将 tokenize 后做单步 prefill）")
    parser.add_argument("--max-layer", type=int, default=None, help="最多测到第几层（含），默认测全部层；可先设 3 或 5 快速试")
    args = parser.parse_args()
    if not args.model or not os.path.isdir(args.model):
        print("请指定有效模型目录: --model /path/to/DeepSeek-R1-Distill-Qwen-1___5B", file=sys.stderr)
        sys.exit(1)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("需要 transformers: pip install transformers", file=sys.stderr)
        sys.exit(1)

    print("加载分词器与 LLAISYS 模型（GPU + 全量 CPU 缓存）...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2(args.model, device=DeviceType.NVIDIA, max_batch_size=1)
    nlayer = model._nlayer

    # 用对话模板得到 "什么是" 的 token 序列（与聊天服务一致）
    prompt = args.prompt
    if hasattr(tokenizer, "apply_chat_template"):
        content = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        token_ids = tokenizer.encode(content, add_special_tokens=True)
    else:
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)

    if not token_ids:
        print("token_ids 为空，请换一个 prompt", file=sys.stderr)
        sys.exit(1)
    print(f"prompt: {prompt!r} -> {len(token_ids)} tokens: {token_ids[:20]}...")

    # 贪心解码，便于对比
    temperature, top_k, top_p, seed = 0.0, 1, 1.0, 0

    max_layer = args.max_layer if args.max_layer is not None else (nlayer - 1)
    if max_layer >= nlayer:
        max_layer = nlayer - 1
    print(f"逐层测试 gpu_up_to_layer from -1 to {max_layer} (共 {max_layer + 2} 次 prefill)...")
    results = []
    baseline = None
    for gpu_up_to_layer in range(-1, max_layer + 1):
        next_tok = model.infer_hybrid(
            token_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            gpu_up_to_layer=gpu_up_to_layer,
        )
        results.append((gpu_up_to_layer, next_tok))
        if gpu_up_to_layer == -1:
            baseline = next_tok
        ok = "✓" if next_tok == baseline else "✗ 与全 CPU 不一致"
        layer_desc = "全 CPU" if gpu_up_to_layer == -1 else f"GPU 到 layer {gpu_up_to_layer} (含 embedding 与 layer 0..{gpu_up_to_layer})"
        print(f"  gpu_up_to_layer={gpu_up_to_layer:3d}  next_token={next_tok:6d}  {layer_desc}  {ok}")

    first_bad = None
    for gpu_up_to_layer, next_tok in results:
        if gpu_up_to_layer >= 0 and next_tok != baseline:
            first_bad = gpu_up_to_layer
            break

    print()
    if first_bad is None:
        print("所有 gpu_up_to_layer 的 next_token 均与全 CPU 一致，未复现问题。可尝试更长 prompt 或不同句子。")
    else:
        print(f"【结论】首个与全 CPU 结果不一致的为 gpu_up_to_layer={first_bad}。")
        if first_bad == 0:
            print("  即：仅 embedding 在 GPU 时结果就错了 -> 问题在 GPU embedding 算子。")
        else:
            print(f"  即：embedding + layer 0..{first_bad-1} 在 GPU 时仍对，加上 layer {first_bad} 在 GPU 后变错 -> 问题在 GPU 第 {first_bad} 层（layer {first_bad}）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
