import argparse
from pathlib import Path

import llaisys
from llaisys.models import Qwen2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="model directory")
    parser.add_argument(
        "--tokenizer",
        required=False,
        type=str,
        help="path to tokenizer.model (defaults to <model>/tokenizer.model)",
    )
    parser.add_argument("--prompt", default="你好", type=str)
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "iluvatar"])
    args = parser.parse_args()

    model_path = Path(args.model)
    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer)
    else:
        tokenizer_path = model_path / "tokenizer.model"
        if not tokenizer_path.exists():
            tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer file not found: {tokenizer_path}")

    tokenizer = llaisys.Tokenizer(str(tokenizer_path))
    model = Qwen2(str(model_path), llaisys.DeviceType.CPU if args.device == "cpu" else llaisys.DeviceType.NVIDIA)

    prompt = Qwen2.build_prompt(
        [{"role": "user", "content": args.prompt}],
        system_prompt="你是助手",
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt)
    output_ids = model.generate(prompt_ids, max_new_tokens=args.max_new_tokens)
    output_text = tokenizer.decode(output_ids)

    print("=== Prompt ===")
    print(prompt)
    print("\n=== Output ===")
    print(output_text)


if __name__ == "__main__":
    main()
