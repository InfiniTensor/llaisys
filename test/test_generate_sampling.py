import argparse
import os

from transformers import AutoTokenizer

import llaisys
from test_utils import llaisys_device


def test_sampling_smoke(model_path: str, device_name: str = "cpu") -> None:
    if not model_path or not os.path.isdir(model_path):
        print("Skipped: provide --model with a local model directory.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    prompt = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": "Say hello in one short sentence."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer.encode(prompt)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=16,
        top_k=20,
        top_p=0.9,
        temperature=0.8,
    )
    generated = output_ids[len(input_ids):]
    assert generated
    assert len(generated) <= 16
    assert all(isinstance(token, int) for token in generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    args = parser.parse_args()

    test_sampling_smoke(args.model, args.device)
    print("\033[92mTest passed!\033[0m\n")
