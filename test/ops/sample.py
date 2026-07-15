import os
import sys

import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
from test_utils import check_equal, llaisys_device, random_tensor, torch_dtype, torch_device


def tensor_from_torch(torch_tensor, device_name="cpu"):
    tensor = llaisys.Tensor(
        torch_tensor.shape,
        dtype=llaisys.DataType.F32,
        device=llaisys_device(device_name),
    )
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(
        tensor.data_ptr(),
        torch_tensor.data_ptr(),
        torch_tensor.numel() * torch_tensor.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return tensor


def allowed_indices(logits: torch.Tensor, top_k: int, top_p: float, temperature: float):
    scaled = logits / temperature
    k = min(top_k, logits.numel())
    values, indices = torch.topk(scaled, k)
    probs = torch.softmax(values, dim=0)
    if top_p >= 1.0:
        return set(indices.tolist())
    cumulative = torch.cumsum(probs, dim=0)
    keep = int((cumulative < top_p).sum().item()) + 1
    return set(indices[:keep].tolist())


def test_top_k_one_is_argmax(device_name="cpu"):
    logits = torch.tensor([0.25, 1.5, -2.0, 0.9], dtype=torch.float32, device=torch_device(device_name))
    token = llaisys.Ops.sample(tensor_from_torch(logits, device_name), top_k=1, top_p=1.0, temperature=1.0)
    assert token == int(torch.argmax(logits).item())


def test_temperature_fallback(device_name="cpu"):
    logits = torch.tensor([0.1, 0.4, 0.3], dtype=torch.float32, device=torch_device(device_name))
    token = llaisys.Ops.sample(tensor_from_torch(logits, device_name), top_k=5, top_p=1.0, temperature=0.0)
    assert token == int(torch.argmax(logits).item())


def test_top_k_membership(device_name="cpu"):
    logits = torch.tensor([0.1, 2.0, 1.5, 3.0], dtype=torch.float32, device=torch_device(device_name))
    allowed = set(torch.topk(logits, 2).indices.tolist())
    tensor = tensor_from_torch(logits, device_name)
    for _ in range(100):
        token = llaisys.Ops.sample(tensor, top_k=2, top_p=1.0, temperature=0.8)
        assert token in allowed


def test_top_p_membership(device_name="cpu"):
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32, device=torch_device(device_name))
    allowed = allowed_indices(logits, top_k=4, top_p=0.7, temperature=0.9)
    tensor = tensor_from_torch(logits, device_name)
    for _ in range(100):
        token = llaisys.Ops.sample(tensor, top_k=4, top_p=0.7, temperature=0.9)
        assert token in allowed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu"], type=str)
    args = parser.parse_args()

    test_top_k_one_is_argmax(args.device)
    test_temperature_fallback(args.device)
    test_top_k_membership(args.device)
    test_top_p_membership(args.device)

    print("\033[92mTest passed!\033[0m\n")
