import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark


def torch_swiglu(out, gate, up):
    torch.mul(up, gate / (1 + torch.exp(-gate.float()).to(out.dtype)), out=out)


def test_op_swiglu(
    shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    gate, gate_ = random_tensor(shape, dtype_name, device_name)
    up, up_ = random_tensor(shape, dtype_name, device_name)

    out, out_ = random_tensor(shape, dtype_name, device_name)
    torch_swiglu(out, gate, up)
    llaisys.Ops.swiglu(out_, gate_, up_)

    assert check_equal(out_, out, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_swiglu(out, gate, up),
            lambda: llaisys.Ops.swiglu(out_, gate_, up_),
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
        # type, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]
    print(f"Testing Ops.swiglu on {args.device}")
    for shape in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_swiglu(shape, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
