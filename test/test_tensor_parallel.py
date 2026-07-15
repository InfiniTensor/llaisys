import argparse
import os

import torch

import llaisys


def test_tensor_parallel_qwen2(
    model_path: str,
    *,
    tp_size: int = 2,
    max_steps: int = 8,
) -> None:
    if not model_path or not os.path.isdir(model_path):
        print("Skipped: provide --model with a local model directory.")
        return
    if not torch.cuda.is_available():
        print("Skipped: CUDA is not available.")
        return
    if torch.cuda.device_count() < tp_size:
        print(f"Skipped: need at least {tp_size} CUDA devices.")
        return

    prompt = [151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198]
    expected_generated = [91786, 0, 358, 2776, 18183, 39350, 10911, 16]
    tp_device_ids = list(range(tp_size))
    parallel = llaisys.models.create_model(
        model_path,
        llaisys.DeviceType.NVIDIA,
        0,
        tp_size=tp_size,
        tp_device_ids=tp_device_ids,
    )

    parallel_out = parallel.generate(
        prompt,
        max_new_tokens=max_steps,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    assert parallel_out[: len(prompt)] == prompt
    assert parallel_out[len(prompt):] == expected_generated[:max_steps]

    close = getattr(parallel, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--tp-size", default=2, type=int)
    parser.add_argument("--max-steps", default=8, type=int)
    args = parser.parse_args()

    test_tensor_parallel_qwen2(
        args.model,
        tp_size=args.tp_size,
        max_steps=args.max_steps,
    )
    print("\033[92mTest passed!\033[0m\n")
