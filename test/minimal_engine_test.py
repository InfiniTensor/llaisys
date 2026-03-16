#!/usr/bin/env python3
"""
最小复现：绕过 FastAPI，直接用 llaisys 引擎 + tokenizer 做一次「完整 prompt prefill + 逐 token 解码」。
用于确认：当首步传入完整 input_ids 时，C++ 推理是否仍输出 1\\n2\\n3 等退化序列。

用法（在项目根目录）:
  PYTHONPATH=. python test/minimal_engine_test.py --model /home/chenncy/llaisys/DeepSeek-R1-Distill-Qwen-1___5B
  PYTHONPATH=. python test/minimal_engine_test.py --model /path/to/model --prompt "什么是数学" --max_steps 50
"""
import argparse
import os
import sys

# 确保能 import 项目内的 llaisys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Minimal llaisys inference test (no FastAPI)")
    parser.add_argument("--model", type=str, required=True, help="Path to model dir (e.g. DeepSeek-R1-Distill-Qwen-1___5B)")
    parser.add_argument("--prompt", type=str, default="你好", help="User message")
    parser.add_argument("--max_steps", type=int, default=30, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "nvidia"])
    args = parser.parse_args()

    from llaisys_py.libllaisys import DeviceType
    from llaisys_py.models.qwen2 import Qwen2

    device = DeviceType.CPU if args.device == "cpu" else DeviceType.NVIDIA
    model_path = os.path.abspath(os.path.expanduser(args.model))
    if not os.path.isdir(model_path):
        print(f"[ERROR] Model path is not a directory: {model_path}", file=sys.stderr)
        sys.exit(1)

    print("[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("[2/3] Loading llaisys Qwen2 model...")
    model = Qwen2(model_path, device=device)

    # 与服务端一致：用 chat template 得到 prompt 字符串，再 encode
    conversation = [{"role": "user", "content": args.prompt}]
    prompt_str = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer.encode(prompt_str)
    print(f"[3/3] Prompt tokens: {len(input_ids)}, first 10 ids: {input_ids[:10]}")

    model.reset_kv_cache()

    # 关键：首步必须传入完整 input_ids，做 prefill；之后每步只传上一个 token
    tokens = list(input_ids)
    full_text = []

    for step in range(args.max_steps):
        if step == 0:
            # Prefill：传入完整 prompt
            next_id = model.next_token(
                tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            )
        else:
            # Decode：只传最后一个 token
            next_id = model.next_token(
                [tokens[-1]],
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            )
        tokens.append(next_id)
        piece = tokenizer.decode([next_id], skip_special_tokens=True)
        full_text.append(piece)
        print(f"  step={step + 1} token_id={next_id} delta={repr(piece)}")
        if next_id == model.end_token:
            print("[EOS]")
            break

    result = "".join(full_text)
    print("\n--- Full generated text ---")
    print(result)
    print("---")
    if not result.strip() or all(c in " \n\t\r0123456789" for c in result.strip()):
        print("\n[WARN] Output looks degenerate (only digits/whitespace). Engine or weights may be broken.")
        sys.exit(1)
    print("\n[OK] Engine produced non-degenerate text.")


if __name__ == "__main__":
    main()
