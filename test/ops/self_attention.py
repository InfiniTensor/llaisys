import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark


def torch_self_attention(attn_val, query, key, value, scale):
    query = query.transpose(-2, -3)
    key = key.transpose(-2, -3)
    value = value.transpose(-2, -3)
    L, S = query.size(-2), key.size(-2)
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=S-L)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)

    key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
    value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_val.copy_((attn_weight @ value).transpose(-2, -3))


def test_op_self_attention(
    qlen,
    kvlen,
    nh,
    nkvh,
    hd,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(
        f"   qlen={qlen} kvlen={kvlen} nh={nh} nkvh={nkvh} hd={hd} dtype <{dtype_name}>"
    )
    q, q_ = random_tensor((qlen, nh, hd), dtype_name, device_name)
    k, k_ = random_tensor((kvlen, nkvh, hd), dtype_name, device_name)
    v, v_ = random_tensor((kvlen, nkvh, hd), dtype_name, device_name)
    scale = 1.0 / (hd**0.5)

    attn_val, attn_val_ = random_tensor((qlen, nh, hd), dtype_name, device_name)
    torch_self_attention(attn_val, q, k, v, scale)
    llaisys.Ops.self_attention(attn_val_, q_, k_, v_, scale)
    assert check_equal(attn_val_, attn_val, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_self_attention(attn_val, q, k, v, scale),
            lambda: llaisys.Ops.self_attention(attn_val_, q_, k_, v_, scale),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [
        # qlen, kvlen, nh, nkvh, hd
        (2, 2, 1, 1, 4),
        (5, 11, 4, 2, 8),
    ]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]
    print(f"Testing Ops.self_attention on {args.device}")
    for shape in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_self_attention(
                *shape, dtype_name, atol, rtol, args.device, args.profile
            )

    print("\033[92mTest passed!\033[0m\n")
