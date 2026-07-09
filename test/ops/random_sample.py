import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch
from test_utils import zero_tensor, check_equal, torch_device


def _torch_to_llaisys(torch_tensor: torch.Tensor, dtype_name: str, device_name: str):
    llaisys_tensor_torch, llaisys_tensor = zero_tensor(torch_tensor.shape, dtype_name, device_name)
    api = llaisys.RuntimeAPI(llaisys.DeviceType.CPU if device_name == "cpu" else llaisys.DeviceType.NVIDIA)
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        torch_tensor.numel() * torch_tensor.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return llaisys_tensor_torch, llaisys_tensor


def _read_scalar_i64(tensor: llaisys.Tensor, device_name: str) -> int:
    out = torch.zeros((1,), dtype=torch.int64, device=torch_device(device_name))
    api = llaisys.RuntimeAPI(tensor.device_type())
    api.memcpy_sync(out.data_ptr(), tensor.data_ptr(), out.numel() * out.element_size(), llaisys.MemcpyKind.D2D)
    return int(out.item())


def _torch_reference_distribution(logits: torch.Tensor, temperature: float, top_k: int, top_p: float):
    scaled_logits = logits.float() / max(temperature, 1e-6)
    sorted_vals, sorted_idx = torch.sort(scaled_logits, descending=True)

    if top_k <= 0 or top_k > sorted_idx.numel():
        top_k = sorted_idx.numel()
    sorted_vals = sorted_vals[:top_k]
    sorted_idx = sorted_idx[:top_k]

    probs = torch.softmax(sorted_vals, dim=0)

    top_p = min(1.0, max(0.0, float(top_p)))
    keep_count = probs.numel()
    if 0.0 < top_p < 1.0:
        cdf = torch.cumsum(probs, dim=0)
        keep_count = int(torch.searchsorted(cdf, torch.tensor(top_p, device=cdf.device), right=False).item()) + 1
        keep_count = max(1, keep_count)

    sorted_idx = sorted_idx[:keep_count]
    probs = probs[:keep_count]
    probs = probs / probs.sum()
    return sorted_idx, probs


def _sample_llaisys_many(logits_: llaisys.Tensor, nsample: int, temperature: float, top_k: int, top_p: float,
                        device_name: str, seed_base: int | None = None):
    idx_t, idx_ = zero_tensor((1,), "i64", device_name)
    val_t, val_ = zero_tensor((1,), "f32", device_name)
    counts = {}

    for _ in range(nsample):
        seed = None if seed_base is None else (seed_base + _)
        llaisys.Ops.random_sample(idx_, val_, logits_, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)
        sampled = _read_scalar_i64(idx_, device_name)
        counts[sampled] = counts.get(sampled, 0) + 1

    return counts


def test_random_sample_topk1_is_argmax(dtype_name="f32", device_name="cpu"):
    print(f"   top_k=1 -> argmax, dtype <{dtype_name}>")
    logits_cpu = torch.tensor([0.1, 3.2, -0.5, 1.1], dtype=torch.float32, device=torch_device(device_name))
    dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_name]
    logits = logits_cpu.to(dtype=dtype)

    _, logits_ = _torch_to_llaisys(logits, dtype_name, device_name)
    max_idx, max_idx_ = zero_tensor((1,), "i64", device_name)
    max_val, max_val_ = zero_tensor((1,), dtype_name, device_name)

    llaisys.Ops.random_sample(max_idx_, max_val_, logits_, temperature=1.0, top_k=1, top_p=1.0)

    expected_idx = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int64)
    expected_val = logits[expected_idx]

    assert check_equal(max_idx_, expected_idx, strict=True)
    assert check_equal(max_val_, expected_val, strict=True)


def test_random_sample_topp0_is_argmax(device_name="cpu"):
    print("   top_p=0 -> argmax fallback")
    logits = torch.tensor([0.2, 1.6, -0.4, 0.7, 1.2], dtype=torch.float32, device=torch_device(device_name))
    _, logits_ = _torch_to_llaisys(logits, "f32", device_name)
    idx_t, idx_ = zero_tensor((1,), "i64", device_name)
    val_t, val_ = zero_tensor((1,), "f32", device_name)

    expected_idx = int(torch.argmax(logits, dim=-1).item())
    for _ in range(64):
        llaisys.Ops.random_sample(idx_, val_, logits_, temperature=1.0, top_k=4, top_p=0.0)
        sampled = _read_scalar_i64(idx_, device_name)
        assert sampled == expected_idx


def test_random_sample_topk_constraint(device_name="cpu"):
    print("   sampled token must be inside torch reference support")
    logits = torch.tensor([0.1, 2.5, -1.0, 1.2, 3.3, 0.9], dtype=torch.float32, device=torch_device(device_name))
    _, logits_ = _torch_to_llaisys(logits, "f32", device_name)

    top_k = 3
    ref_idx, _ = _torch_reference_distribution(logits, temperature=1.0, top_k=top_k, top_p=1.0)
    ref_set = set(ref_idx.tolist())

    counts = _sample_llaisys_many(logits_, nsample=256, temperature=1.0, top_k=top_k, top_p=1.0, device_name=device_name)
    for sampled in counts:
        assert sampled in ref_set


def test_random_sample_topp_constraint(device_name="cpu"):
    print("   sampling distribution should match torch reference")
    logits = torch.tensor([2.0, 1.2, 0.7, -0.5, -1.3], dtype=torch.float32, device=torch_device(device_name))
    _, logits_ = _torch_to_llaisys(logits, "f32", device_name)
    temperature = 0.9
    top_k = 5
    top_p = 0.8
    nsample = 4000

    ref_idx, ref_prob = _torch_reference_distribution(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    counts = _sample_llaisys_many(logits_, nsample=nsample, temperature=temperature, top_k=top_k, top_p=top_p, device_name=device_name)

    for sampled in counts:
        assert sampled in set(ref_idx.tolist())

    empirical = []
    for token in ref_idx.tolist():
        empirical.append(counts.get(token, 0) / nsample)

    empirical = torch.tensor(empirical, dtype=torch.float32, device=ref_prob.device)
    max_diff = torch.max(torch.abs(empirical - ref_prob)).item()
    assert max_diff < 0.01, f"distribution mismatch too large: {max_diff}"


def test_random_sample_seed_reproducibility(device_name="cpu"):
    print("   seeded sampling should be reproducible")
    logits = torch.tensor([2.0, 1.1, 0.8, -0.3, -1.8], dtype=torch.float32, device=torch_device(device_name))
    _, logits_ = _torch_to_llaisys(logits, "f32", device_name)

    counts_run1 = _sample_llaisys_many(
        logits_,
        nsample=256,
        temperature=0.95,
        top_k=5,
        top_p=0.9,
        device_name=device_name,
        seed_base=12345,
    )
    counts_run2 = _sample_llaisys_many(
        logits_,
        nsample=256,
        temperature=0.95,
        top_k=5,
        top_p=0.9,
        device_name=device_name,
        seed_base=12345,
    )
    assert counts_run1 == counts_run2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()

    print(f"Testing Ops.random_sample on {args.device}")
    for dtype_name in ["f32", "f16", "bf16"]:
        test_random_sample_topk1_is_argmax(dtype_name=dtype_name, device_name=args.device)

    test_random_sample_topp0_is_argmax(device_name=args.device)
    test_random_sample_topk_constraint(device_name=args.device)
    test_random_sample_topp_constraint(device_name=args.device)
    test_random_sample_seed_reproducibility(device_name=args.device)

    print("\033[92mTest passed!\033[0m\n")
