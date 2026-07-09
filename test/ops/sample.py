import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
import numpy as np
from test_utils import random_tensor, check_equal, zero_tensor


# ── helpers ──────────────────────────────────────────────────────────────────

def make_logits_ll(logits_np, dtype_name):
    dtype_map = {
        "f32": (torch.float32, llaisys.DataType.F32),
        "f16": (torch.float16, llaisys.DataType.F16),
        "bf16": (torch.bfloat16, llaisys.DataType.BF16),
    }
    td, ld = dtype_map[dtype_name]
    t = torch.tensor(logits_np, dtype=td)
    ll = llaisys.Tensor(logits_np.shape, dtype=ld, device=llaisys.DeviceType.CPU)
    api = llaisys.RuntimeAPI(llaisys.DeviceType.CPU)
    api.memcpy_sync(ll.data_ptr(), t.data_ptr(), t.numel() * t.element_size(), llaisys.MemcpyKind.D2D)
    return ll


def run_sample(logits_np, dtype_name="f32", temperature=1.0, top_k=0, top_p=1.0):
    ll = make_logits_ll(logits_np, dtype_name)
    out = llaisys.Tensor((1,), dtype=llaisys.DataType.I64, device=llaisys.DeviceType.CPU)
    z = torch.zeros((1,), dtype=torch.int64)
    api = llaisys.RuntimeAPI(llaisys.DeviceType.CPU)
    api.memcpy_sync(out.data_ptr(), z.data_ptr(), 8, llaisys.MemcpyKind.D2D)
    llaisys.Ops.sample(out, ll, temperature, top_k, top_p)
    r = torch.zeros((1,), dtype=torch.int64)
    api.memcpy_sync(r.data_ptr(), out.data_ptr(), 8, llaisys.MemcpyKind.D2D)
    return int(r[0])


# ── tests ─────────────────────────────────────────────────────────────────────

def test_op_sample_temperature(device_name="cpu", profile=False):
    """temperature scaling: top_k=1 always returns argmax regardless of temperature."""
    print("   [temperature] top_k=1 always returns argmax")
    logits = np.array([1.0, 5.0, 2.0, 3.0], dtype=np.float32)
    for temp in [0.5, 1.0, 2.0]:
        for dtype in ["f32", "f16", "bf16"]:
            results = [run_sample(logits, dtype_name=dtype, temperature=temp, top_k=1) for _ in range(10)]
            assert all(r == 1 for r in results), f"temperature={temp} dtype={dtype} failed: {results}"


def test_op_sample_distribution(device_name="cpu", profile=False):
    """No filtering: sampled distribution should match softmax probabilities."""
    print("   [distribution] multinomial sampling matches softmax probs")
    logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    expected_probs = torch.softmax(torch.tensor(logits), dim=0).numpy()

    n = 5000
    counts = np.zeros(4)
    for _ in range(n):
        counts[run_sample(logits, top_k=0, top_p=1.0)] += 1
    observed = counts / n

    # Allow 5% absolute tolerance for statistical test
    for i in range(4):
        assert abs(observed[i] - expected_probs[i]) < 0.05, (
            f"token {i}: observed={observed[i]:.3f} expected={expected_probs[i]:.3f}"
        )


def test_op_sample_top_k(device_name="cpu", profile=False):
    """top_k=K: only top-K tokens should ever be sampled."""
    print("   [top_k] only top-K tokens sampled")
    logits = np.array([1.0, 5.0, 4.0, 0.5], dtype=np.float32)  # sorted: idx 1 > 2 > 0 > 3

    # top_k=1 -> always argmax
    results = [run_sample(logits, top_k=1) for _ in range(50)]
    assert all(r == 1 for r in results), f"top_k=1 failed: {set(results)}"

    # top_k=2 -> only idx 1 and 2
    counts = [0] * 4
    for _ in range(1000):
        counts[run_sample(logits, top_k=2)] += 1
    assert counts[0] == 0 and counts[3] == 0, f"top_k=2 leaked outside top-2: {counts}"
    assert counts[1] > 0 and counts[2] > 0, f"top_k=2 missing expected tokens: {counts}"


def test_op_sample_top_p(device_name="cpu", profile=False):
    """top_p filtering: very low top_p should collapse to argmax."""
    print("   [top_p] top_p=0.0 always returns argmax")
    logits = np.array([1.0, 10.0, 2.0, 3.0], dtype=np.float32)
    results = [run_sample(logits, top_p=0.0) for _ in range(30)]
    assert all(r == 1 for r in results), f"top_p=0.0 failed: {set(results)}"

    # top_p=1.0 should allow all tokens (with enough samples, non-argmax tokens appear)
    print("   [top_p] top_p=1.0 allows non-argmax tokens")
    logits2 = np.array([3.0, 4.0, 3.0, 3.0], dtype=np.float32)
    seen = set()
    for _ in range(500):
        seen.add(run_sample(logits2, top_p=1.0))
    assert len(seen) > 1, f"top_p=1.0 only sampled {seen}"


def test_op_sample_dtype(device_name="cpu", profile=False):
    """f16 and bf16 logits: top_k=1 should still return correct argmax."""
    print("   [dtype] f16/bf16 logits return correct argmax with top_k=1")
    logits = np.array([1.0, 10.0, 2.0, 3.0], dtype=np.float32)
    for dtype in ["f16", "bf16"]:
        results = [run_sample(logits, dtype_name=dtype, top_k=1) for _ in range(20)]
        assert all(r == 1 for r in results), f"{dtype} top_k=1 failed: {set(results)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    print(f"Testing Ops.sample on {args.device}")
    test_op_sample_temperature(args.device, args.profile)
    test_op_sample_distribution(args.device, args.profile)
    test_op_sample_top_k(args.device, args.profile)
    test_op_sample_top_p(args.device, args.profile)
    test_op_sample_dtype(args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
