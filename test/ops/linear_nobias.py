import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch
import llaisys

from llaisys.libllaisys import LIB_LLAISYS
from test_utils import random_tensor, check_equal, benchmark


def torch_linear_nobias(out, x, w):
    # out = x @ w.T
    torch.nn.functional.linear(x, w, bias=None, out=out)


def llaisys_linear_nobias(out_, x_, w_):
    # Call the C API directly so we can pass NULL bias.
    LIB_LLAISYS.llaisysLinear(out_.lib_tensor(), x_.lib_tensor(), w_.lib_tensor(), None)


def test_op_linear_nobias(
    out_shape,
    x_shape,
    w_shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   out {out_shape}, x {x_shape}, w {w_shape}, bias False, dtype <{dtype_name}>")
    x, x_ = random_tensor(x_shape, dtype_name, device_name, scale=0.1)
    w, w_ = random_tensor(w_shape, dtype_name, device_name, scale=0.01)

    out, out_ = random_tensor(out_shape, dtype_name, device_name)
    torch_linear_nobias(out, x, w)
    llaisys_linear_nobias(out_, x_, w_)

    assert check_equal(out_, out, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_linear_nobias(out, x, w),
            lambda: llaisys_linear_nobias(out_, x_, w_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    test_shapes = [
        ((2, 3), (2, 4), (3, 4)),
        ((8, 16), (8, 32), (16, 32)),
        ((512, 4096), (512, 4096), (4096, 4096)),
    ]
    test_dtype_prec = [
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    print(f"Testing Ops.linear (no bias) on {args.device}")
    for shapes in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_linear_nobias(*shapes, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")

