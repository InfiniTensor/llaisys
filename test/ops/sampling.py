import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch
import llaisys


def _make_logits_tensor(logits: torch.Tensor, device_name: str):
    t = llaisys.Tensor(
        logits.shape,
        dtype=llaisys.DataType.F32,
        device=llaisys.DeviceType.NVIDIA if device_name == "nvidia" else llaisys.DeviceType.CPU,
        device_id=0,
    )
    api = llaisys.RuntimeAPI(t.device_type())
    api.memcpy_sync(t.data_ptr(), logits.data_ptr(), logits.numel() * logits.element_size(), llaisys.MemcpyKind.D2D)
    return t


def _read_i64_scalar(t: llaisys.Tensor) -> int:
    out = torch.zeros((1,), dtype=torch.int64, device=torch.device("cuda" if t.device_type() == llaisys.DeviceType.NVIDIA else "cpu"))
    api = llaisys.RuntimeAPI(t.device_type())
    api.memcpy_sync(out.data_ptr(), t.data_ptr(), out.numel() * out.element_size(), llaisys.MemcpyKind.D2D)
    return int(out.item())


def test_sampling(device_name: str):
    print(f"Testing Ops.sample on {device_name}")
    device = torch.device("cuda" if device_name == "nvidia" else "cpu")
    logits = torch.tensor([0.1, 2.0, 0.5, 1.0], dtype=torch.float32, device=device)

    logits_t = _make_logits_tensor(logits, device_name)
    out_idx = llaisys.Tensor((1,), dtype=llaisys.DataType.I64,
                             device=llaisys.DeviceType.NVIDIA if device_name == "nvidia" else llaisys.DeviceType.CPU,
                             device_id=0)

    # top_k=1 should behave like argmax.
    llaisys.Ops.sample(out_idx, logits_t, temperature=1.0, top_k=1, top_p=1.0)
    idx = _read_i64_scalar(out_idx)
    assert idx == 1, f"Expected argmax index 1, got {idx}"

    # For top_k=2, sampled index should always be one of top-2 entries.
    allowed = {1, 3}
    for _ in range(64):
        llaisys.Ops.sample(out_idx, logits_t, temperature=0.9, top_k=2, top_p=1.0)
        idx = _read_i64_scalar(out_idx)
        assert idx in allowed, f"Expected sampled idx in {allowed}, got {idx}"

    print("\033[92mTest passed!\033[0m\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()
    test_sampling(args.device)
