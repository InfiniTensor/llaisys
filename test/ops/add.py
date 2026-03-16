import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys_py
import torch
from test_utils import random_tensor, check_equal, benchmark


def torch_add(ans, a, b):
    torch.add(a, b, out=ans)


def test_op_add(
    shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    a, a_ = random_tensor(shape, dtype_name, device_name)
    b, b_ = random_tensor(shape, dtype_name, device_name)

    c, c_ = random_tensor(shape, dtype_name, device_name)
    torch_add(c, a, b)
    llaisys_py.Ops.add(c_, a_, b_)

    assert check_equal(c_, c, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_add(c, a, b),
            lambda: llaisys_py.Ops.add(c_, a_, b_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [(2, 3), (512, 4096)]
    testDtypePrec = [
        # type, atol, rtol（bf16 用 1e-2 以兼容 float 累加再舍入与 PyTorch 原生 bf16 的细微差异）
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]
    print(f"Testing Ops.add on {args.device}")
    for shape in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_add(shape, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
