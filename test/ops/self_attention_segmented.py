import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch
import llaisys
from test_utils import random_tensor, check_equal


def torch_self_attention_segmented(attn_val, query, key, value, scale, q_offsets, kv_offsets):
    # query/key/value: [seq, head, dim]
    q = query.transpose(-2, -3)  # [head, qlen, dim]
    k = key.transpose(-2, -3)    # [kv_head, kvlen, dim]
    v = value.transpose(-2, -3)  # [kv_head, kvlen, dim]

    nhead = q.size(0)
    nkvh = k.size(0)
    rep = nhead // nkvh
    k = k.repeat_interleave(rep, dim=0)
    v = v.repeat_interleave(rep, dim=0)

    qlen = q.size(1)
    kvlen = k.size(1)
    logits = (q @ k.transpose(-2, -1)) * scale  # [head, qlen, kvlen]
    bias = torch.full((qlen, kvlen), float("-inf"), dtype=logits.dtype, device=logits.device)

    for seg in range(len(q_offsets) - 1):
        qb, qe = int(q_offsets[seg]), int(q_offsets[seg + 1])
        kb, ke = int(kv_offsets[seg]), int(kv_offsets[seg + 1])
        seg_qlen = qe - qb
        seg_kvlen = ke - kb
        for s in range(seg_qlen):
            allow = kb + s + (seg_kvlen - seg_qlen)
            bias[qb + s, kb : allow + 1] = 0.0

    logits = logits + bias.unsqueeze(0)
    probs = torch.softmax(logits, dim=-1)
    out = (probs @ v).transpose(-2, -3)  # [qlen, head, dim]
    attn_val.copy_(out)


def test_op_self_attention_segmented(dtype_name="f32", atol=1e-5, rtol=1e-5, device_name="cpu"):
    q_offsets = [0, 2, 3]
    kv_offsets = [0, 4, 6]
    qlen = q_offsets[-1]
    kvlen = kv_offsets[-1]
    nh = 4
    nkvh = 2
    hd = 8

    q, q_ = random_tensor((qlen, nh, hd), dtype_name, device_name)
    k, k_ = random_tensor((kvlen, nkvh, hd), dtype_name, device_name)
    v, v_ = random_tensor((kvlen, nkvh, hd), dtype_name, device_name)
    scale = 1.0 / (hd ** 0.5)

    attn_val, attn_val_ = random_tensor((qlen, nh, hd), dtype_name, device_name)
    torch_self_attention_segmented(attn_val, q, k, v, scale, q_offsets, kv_offsets)
    llaisys.Ops.self_attention_segmented(attn_val_, q_, k_, v_, scale, q_offsets, kv_offsets)
    assert check_equal(attn_val_, attn_val, atol=atol, rtol=rtol)


if __name__ == "__main__":
    print("Testing Ops.self_attention_segmented on cpu")
    test_op_self_attention_segmented("f32", 1e-5, 1e-5, "cpu")
    test_op_self_attention_segmented("f16", 1e-3, 1e-3, "cpu")
    test_op_self_attention_segmented("bf16", 1e-2, 1e-2, "cpu")
    print("\033[92mTest passed!\033[0m\n")
