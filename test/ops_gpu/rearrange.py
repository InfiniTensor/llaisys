import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, llaisys_dtype, llaisys_device


def torch_rearrange(out, x):
    out.copy_(x)


def test_op_rearrange(
    shape,
    dtype_name="f32",
    device_name="nvidia",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    x, x_ = random_tensor(shape, dtype_name, device_name)
    x_perm = x.permute(1, 0)
    x_perm_ = x_.permute(1, 0)

    out = x_perm.contiguous()
    out_ = llaisys.Tensor(out.shape, dtype=llaisys_dtype(dtype_name), device=llaisys_device(device_name))
    torch_rearrange(out, x_perm)
    llaisys.Ops.rearrange(out_, x_perm_)

    assert check_equal(out_, out, strict=True)

    if profile:
        benchmark(
            lambda: torch_rearrange(out, x_perm),
            lambda: llaisys.Ops.rearrange(out_, x_perm_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia", "iluvatar"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [(2, 3), (16, 64)]
    testDtype = ["f32", "f16", "bf16"]
    print(f"Testing Ops.rearrange on {args.device}")
    for shape in testShapes:
        for dtype_name in testDtype:
            test_op_rearrange(shape, dtype_name, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
