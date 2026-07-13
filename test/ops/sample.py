import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch
from test_utils import random_tensor, zero_tensor, check_equal, benchmark


# ─── helpers ────────────────────────────────────────────────────────────────

def _read_i64(tensor: llaisys.Tensor) -> int:
    """Copy a single-element int64 tensor to host and return its value."""
    tmp = torch.zeros((1,), dtype=torch.int64)
    api = llaisys.RuntimeAPI(tensor.device_type())
    api.memcpy_sync(
        tmp.data_ptr(), tensor.data_ptr(),
        tmp.numel() * tmp.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return int(tmp.item())


def _make_logits(values: list[float], dtype_name: str, device_name: str):
    """Build a 1-D logits tensor from explicit values."""
    t = torch.tensor(values, dtype=_torch_float_dtype(dtype_name))
    ls = llaisys.Tensor((len(values),), dtype=llaisys_dtype(dtype_name),
                        device=llaisys_device(device_name))
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(ls.data_ptr(), t.data_ptr(),
                    t.numel() * t.element_size(), llaisys.MemcpyKind.D2D)
    return t, ls


# ─── dtype helpers ──────────────────────────────────────────────────────────

def _torch_float_dtype(dtype_name: str):
    return {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]


def llaisys_dtype(dtype_name: str):
    return {"f32": llaisys.DataType.F32, "f16": llaisys.DataType.F16,
            "bf16": llaisys.DataType.BF16}[dtype_name]


def llaisys_device(device_name: str):
    return {"cpu": llaisys.DeviceType.CPU, "nvidia": llaisys.DeviceType.NVIDIA}[device_name]


# ─── test cases ─────────────────────────────────────────────────────────────

def test_argmax_mode(vocab_size: int, dtype_name: str, device_name: str):
    """top_k=1 must return the argmax token deterministically."""
    print(f"   [argmax mode] vocab={vocab_size} dtype=<{dtype_name}>")
    logits_t, logits_ls = random_tensor((vocab_size,), dtype_name, device_name)

    out_ls = llaisys.Tensor((1,), dtype=llaisys.DataType.I64,
                            device=llaisys_device(device_name))

    # Run 10 times – result must always be the argmax.
    expected = int(logits_t.argmax().item())
    for _ in range(10):
        llaisys.Ops.sample(out_ls, logits_ls, top_k=1, top_p=1.0, temperature=1.0)
        assert _read_i64(out_ls) == expected, \
            f"top_k=1 should always return argmax {expected}, got {_read_i64(out_ls)}"


def test_topk_constraint(vocab_size: int, top_k: int, dtype_name: str, device_name: str):
    """Sampled token must be within the top-k logit indices."""
    print(f"   [top-k] vocab={vocab_size} top_k={top_k} dtype=<{dtype_name}>")
    logits_t, logits_ls = random_tensor((vocab_size,), dtype_name, device_name)

    # For low-precision dtypes (especially bf16), ties at the k-th boundary are common.
    # Treat all tokens whose logit is >= the k-th largest value as valid top-k candidates.
    logits_f = logits_t.float()
    kth_value = torch.topk(logits_f, top_k).values[-1]
    allowed = set((logits_f >= kth_value).nonzero(as_tuple=False).view(-1).tolist())

    out_ls = llaisys.Tensor((1,), dtype=llaisys.DataType.I64,
                            device=llaisys_device(device_name))

    llaisys.Ops.sample_set_seed(42)
    for _ in range(50):
        llaisys.Ops.sample(out_ls, logits_ls, top_k=top_k, top_p=1.0, temperature=1.0)
        token = _read_i64(out_ls)
        assert token in allowed, \
            f"top_k={top_k}: sampled token {token} not in top-k set {allowed}"


def test_topp_constraint(vocab_size: int, top_p: float, dtype_name: str, device_name: str):
    """Sampled token must lie within the nucleus (top-p) set."""
    print(f"   [top-p] vocab={vocab_size} top_p={top_p} dtype=<{dtype_name}>")
    logits_t, logits_ls = random_tensor((vocab_size,), dtype_name, device_name)

    # Compute nucleus on the torch side.
    probs = torch.softmax(logits_t.float(), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    # All tokens in the nucleus: those at or before the first cumsum >= top_p.
    cutoff = int((cumsum < top_p).sum().item()) + 1
    # Include ties on the cutoff probability to avoid false negatives in low precision.
    cutoff_prob = float(sorted_probs[cutoff - 1].item())
    nucleus = set((probs >= cutoff_prob).nonzero(as_tuple=False).view(-1).tolist())

    out_ls = llaisys.Tensor((1,), dtype=llaisys.DataType.I64,
                            device=llaisys_device(device_name))

    llaisys.Ops.sample_set_seed(0)
    for _ in range(50):
        llaisys.Ops.sample(out_ls, logits_ls, top_k=0, top_p=top_p, temperature=1.0)
        token = _read_i64(out_ls)
        assert token in nucleus, \
            f"top_p={top_p}: sampled token {token} not in nucleus {nucleus}"


def test_temperature_distribution(vocab_size: int, dtype_name: str, device_name: str):
    """
    With high temperature and many draws, the empirical distribution should
    be roughly uniform – no single token dominates.
    With temperature=0.01 it should converge to near-argmax behaviour.
    We do a light sanity check rather than a full statistical test.
    """
    print(f"   [temperature] vocab={vocab_size} dtype=<{dtype_name}>")
    logits_t, logits_ls = random_tensor((vocab_size,), dtype_name, device_name)

    out_ls = llaisys.Tensor((1,), dtype=llaisys.DataType.I64,
                            device=llaisys_device(device_name))

    # Very low temperature → all draws should be same token (argmax).
    argmax_token = int(logits_t.float().argmax().item())
    llaisys.Ops.sample_set_seed(7)
    for _ in range(20):
        llaisys.Ops.sample(out_ls, logits_ls, top_k=0, top_p=1.0, temperature=1e-6)
        assert _read_i64(out_ls) == argmax_token, \
            "temperature≈0 should always sample the argmax token"

    # Very high temperature → diversity check: over 200 draws at least 2 distinct tokens
    if vocab_size >= 4:
        llaisys.Ops.sample_set_seed(13)
        seen = set()
        for _ in range(200):
            llaisys.Ops.sample(out_ls, logits_ls, top_k=0, top_p=1.0, temperature=1e6)
            seen.add(_read_i64(out_ls))
        assert len(seen) >= 2, \
            "temperature=1e6 (near-uniform) should sample diverse tokens"


def test_seed_reproducibility(vocab_size: int, dtype_name: str, device_name: str):
    """Same seed must produce the same sequence of sampled tokens."""
    print(f"   [seed reproducibility] vocab={vocab_size} dtype=<{dtype_name}>")
    logits_t, logits_ls = random_tensor((vocab_size,), dtype_name, device_name)
    out_ls = llaisys.Tensor((1,), dtype=llaisys.DataType.I64,
                            device=llaisys_device(device_name))

    llaisys.Ops.sample_set_seed(99)
    run1 = []
    for _ in range(20):
        llaisys.Ops.sample(out_ls, logits_ls, top_k=0, top_p=1.0, temperature=1.0)
        run1.append(_read_i64(out_ls))

    llaisys.Ops.sample_set_seed(99)
    run2 = []
    for _ in range(20):
        llaisys.Ops.sample(out_ls, logits_ls, top_k=0, top_p=1.0, temperature=1.0)
        run2.append(_read_i64(out_ls))

    assert run1 == run2, f"Same seed must give same sequence.\nrun1={run1}\nrun2={run2}"


def test_profile(vocab_size: int, dtype_name: str, device_name: str):
    logits_t, logits_ls = random_tensor((vocab_size,), dtype_name, device_name)
    out_ls = llaisys.Tensor((1,), dtype=llaisys.DataType.I64,
                            device=llaisys_device(device_name))

    def torch_sample():
        probs = torch.softmax(logits_t.float(), dim=-1)
        torch.multinomial(probs, 1)

    benchmark(
        torch_sample,
        lambda: llaisys.Ops.sample(out_ls, logits_ls, top_k=50, top_p=0.9, temperature=0.8),
        device_name,
    )


# ─── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    DTYPES = ["f32", "f16", "bf16"]
    VOCAB_SIZES = [16, 256, 4096]

    print(f"Testing Ops.sample on {args.device}")

    for vocab in VOCAB_SIZES:
        for dtype in DTYPES:
            test_argmax_mode(vocab, dtype, args.device)

    for vocab in VOCAB_SIZES:
        for dtype in DTYPES:
            test_topk_constraint(vocab, top_k=min(5, vocab // 2), dtype_name=dtype, device_name=args.device)

    for vocab in VOCAB_SIZES:
        for dtype in DTYPES:
            test_topp_constraint(vocab, top_p=0.9, dtype_name=dtype, device_name=args.device)

    for vocab in VOCAB_SIZES:
        for dtype in DTYPES:
            test_temperature_distribution(vocab, dtype, args.device)

    for vocab in VOCAB_SIZES:
        for dtype in DTYPES:
            test_seed_reproducibility(vocab, dtype, args.device)

    if args.profile:
        print("\nBenchmark (vocab=32000, f32):")
        test_profile(32000, "f32", args.device)

    print("\033[92mTest passed!\033[0m\n")
