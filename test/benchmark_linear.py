import sys
import os
import torch
import torch.utils.benchmark as benchmark

# Add project root to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
from test.test_utils import random_tensor, llaisys_device

def run_benchmark(M, K, N, dtype="f32"):
    """
    Benchmark Linear operator using torch.utils.benchmark
    M: Batch size
    K: Input features
    N: Output features
    """
    label = "Linear Operator"
    sub_label = f"M={M}, K={K}, N={N}, dtype={dtype}"
    device = "cpu"
    
    # Prepare data
    # Input X: [M, K]
    x, x_llai = random_tensor((M, K), dtype, device)
    # Weight W: [N, K] (PyTorch Linear weights are [out_features, in_features])
    w, w_llai = random_tensor((N, K), dtype, device)
    # Bias: [N]
    b, b_llai = random_tensor((N,), dtype, device)
    
    # Output placeholder
    out_llai = llaisys.Tensor(
        (M, N),
        dtype=x_llai.dtype(),
        device=llaisys_device(device),
        device_id=0
    )
    
    # Get current number of threads to ensure fair comparison
    num_threads = torch.get_num_threads()
    
    # Task 1: PyTorch (Warmup/Baseline only)
    # ... code removed ...
    
    print(f"Benchmarking LLAISYS Linear: M={M}, K={K}, N={N}, dtype={dtype}")

    # Task 2: LLAISYS
    t1 = benchmark.Timer(
        stmt='llaisys.Ops.linear(out, x, w, b)',
        globals={'llaisys': llaisys, 'out': out_llai, 'x': x_llai, 'w': w_llai, 'b': b_llai},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description="LLAISYS",
    )
    # Increase min_run_time to ensure enough repetitions for small workloads
    # 5.0s is usually enough to get stable results even for microsecond-level ops
    m = t1.blocked_autorange(min_run_time=5.0)
    print(f"  Result: {m.mean * 1000:.4f} ms (median: {m.median * 1000:.4f} ms, iqr: {m.iqr * 1000:.4f} ms)")

if __name__ == "__main__":
    print(f"Running benchmarks with {torch.get_num_threads()} threads...\n")
    
    # Small scale
    run_benchmark(32, 64, 32)
    
    # Medium scale (Simulate typical hidden layers)
    # e.g., Batch=1, Seq=128, Hidden=4096 -> MLP expansion
    run_benchmark(128, 4096, 4096)
