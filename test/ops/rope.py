import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import arrange_tensor, random_tensor, check_equal, benchmark


def torch_rope(y: torch.Tensor, x: torch.Tensor, pos_ids: torch.Tensor, theta: float):
    assert y.dim() == 3
    seq_len, n_heads, head_dim = y.shape
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE."

    # Split into [a, b] pairs
    x_a, x_b = x[..., : head_dim // 2], x[..., head_dim // 2 :]

    # [seq_len] positions starting from start_pos
    positions = pos_ids.to(torch.float32).unsqueeze(1)  # [seq_len, 1]

    # RoPE frequency exponents: 1 / theta^(2i / d)
    i = torch.arange(0, head_dim // 2, dtype=torch.float32, device=y.device)  # [1, head_dim//2]
    freqs = positions / (theta ** (2 * i / head_dim))  # [seq_len, head_dim//2]

    sin, cos = freqs.sin(), freqs.cos()
    sin = sin.unsqueeze(1)  # [seq_len, 1, dim/2]
    cos = cos.unsqueeze(1)

    # Apply rotation
    y[..., : head_dim // 2] = x_a * cos - x_b * sin
    y[..., head_dim // 2 :] = x_b * cos + x_a * sin


def test_op_rope(
    shape,
    start_end,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} range {start_end} dtype <{dtype_name}>")
    x, x_ = random_tensor(shape, dtype_name, device_name)
    pos_ids, pos_ids_ = arrange_tensor(start_end[0], start_end[1], device_name)
    theta = 10000.0
    y, y_ = random_tensor(shape, dtype_name, device_name)
    torch_rope(y, x, pos_ids, theta)
    llaisys.Ops.rope(y_, x_, pos_ids_, theta)

    assert check_equal(y_, y, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_rope(y, x, pos_ids, theta),
            lambda: llaisys.Ops.rope(y_, x_, pos_ids_, theta),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [
        ((2, 1, 4), (0, 2)), 
        ((512, 4, 4096), (512, 1024))]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 1e-4, 1e-4),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]
    print(f"Testing Ops.rope on {args.device}")
    for shape, start_end in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_rope(shape, start_end, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
