import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch

from test_utils import random_tensor, check_equal, benchmark


def torch_rearrange(out, x, perm):
    out.copy_(x.permute(*perm).contiguous())


def llaisys_rearrange(out_, x_, perm):
    # Create a non-contiguous view via permute, then rearrange into a contiguous output.
    x_view_ = x_.permute(*perm)
    llaisys.Ops.rearrange(out_, x_view_)


def test_op_rearrange(
    shape,
    perm,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} perm {perm} dtype <{dtype_name}>")
    x, x_ = random_tensor(shape, dtype_name, device_name, scale=0.1)

    out_shape = tuple(shape[p] for p in perm)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    out_ = llaisys.Tensor(out_shape, dtype=llaisys_dtype(dtype_name), device=llaisys_device(device_name))

    torch_rearrange(out, x, perm)
    llaisys_rearrange(out_, x_, perm)

    assert check_equal(out_, out, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_rearrange(out, x, perm),
            lambda: llaisys_rearrange(out_, x_, perm),
            device_name,
        )


def llaisys_device(device_name: str):
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    elif device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    raise ValueError(device_name)


def llaisys_dtype(dtype_name: str):
    if dtype_name == "f32":
        return llaisys.DataType.F32
    if dtype_name == "f16":
        return llaisys.DataType.F16
    if dtype_name == "bf16":
        return llaisys.DataType.BF16
    raise ValueError(dtype_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    test_shapes = [
        ((2, 3, 4), (2, 0, 1)),
        ((4, 5, 6), (1, 2, 0)),
    ]
    test_dtype_prec = [
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    print(f"Testing Ops.rearrange on {args.device}")
    for shape, perm in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_rearrange(shape, perm, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")

