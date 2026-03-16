"""
可复现的 linear 性能 benchmark，用于优化前后对比。
用法见 docs/cpu-inference-optimization.md 第 5.5 节。
"""
import sys
import os
import time
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys_py
import torch
from test_utils import random_tensor, check_equal, llaisys_device


def torch_linear(out, x, w, bias):
    torch.nn.functional.linear(x, w, bias, out=out)


# 与 test/ops/linear.py 一致的 shape 列表
BENCH_SHAPES = {
    "small": [((2, 3), (2, 4), (3, 4), True)],
    "large": [((512, 4096), (512, 4096), (4096, 4096), True)],
    "all": [
        ((2, 3), (2, 4), (3, 4), True),
        ((512, 4096), (512, 4096), (4096, 4096), True),
    ],
}

DTYPE_TOL = {
    "f32": (1e-5, 1e-5),
    "f16": (1e-3, 1e-3),
    "bf16": (1e-2, 1e-2),
}


def run_one_bench(out_shape, x_shape, w_shape, use_bias, dtype_name, device_name, warmup, repeat):
    atol, rtol = DTYPE_TOL.get(dtype_name, (1e-5, 1e-5))
    x, x_ = random_tensor(x_shape, dtype_name, device_name, scale=0.1)
    w, w_ = random_tensor(w_shape, dtype_name, device_name, scale=0.01)
    bias, bias_ = None, None
    if use_bias:
        bias, bias_ = random_tensor((w_shape[0],), dtype_name, device_name)
    out, out_ = random_tensor(out_shape, dtype_name, device_name)

    # 正确性
    torch_linear(out, x, w, bias)
    llaisys_py.Ops.linear(out_, x_, w_, bias_)
    assert check_equal(out_, out, atol=atol, rtol=rtol), f"check_equal failed {out_shape} {dtype_name}"

    api = llaisys_py.RuntimeAPI(llaisys_device(device_name))

    def time_op(func):
        for _ in range(warmup):
            func()
        api.device_synchronize()
        start = time.perf_counter()
        for _ in range(repeat):
            func()
        api.device_synchronize()
        return (time.perf_counter() - start) / repeat * 1000.0  # ms

    torch_ms = time_op(lambda: torch_linear(out, x, w, bias))
    lla_ms = time_op(lambda: llaisys_py.Ops.linear(out_, x_, w_, bias_))
    return torch_ms, lla_ms


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Linear op benchmark for before/after optimization")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--dtype", default="f32", choices=["f32", "f16", "bf16", "all"],
                        help="dtype to benchmark; 'all' runs f32, f16, bf16")
    parser.add_argument("--shape", default="large", choices=["small", "large", "all"],
                        help="shape set: large=(512,4096)@(4096,4096), small=(2,3)@(3,4), all=both")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="output JSON for baseline/optimized diff")
    args = parser.parse_args()

    shapes_list = BENCH_SHAPES[args.shape]
    dtypes = ["f32", "f16", "bf16"] if args.dtype == "all" else [args.dtype]

    config = {
        "device": args.device,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "shape_set": args.shape,
        "dtype": args.dtype,
    }
    results = []

    for (out_shape, x_shape, w_shape, use_bias) in shapes_list:
        for dtype_name in dtypes:
            torch_ms, lla_ms = run_one_bench(
                out_shape, x_shape, w_shape, use_bias, dtype_name, args.device, args.warmup, args.repeat
            )
            results.append({
                "out_shape": list(out_shape),
                "x_shape": list(x_shape),
                "w_shape": list(w_shape),
                "dtype": dtype_name,
                "torch_ms": round(torch_ms, 4),
                "lla_ms": round(lla_ms, 4),
            })
            if not args.json:
                print(f"   out {out_shape}, x {x_shape}, w {w_shape}, dtype {dtype_name}")
                print(f"        Torch: {torch_ms:.4f} ms   LLAISYS: {lla_ms:.4f} ms")

    if args.json:
        out = {"config": config, "results": results}
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
