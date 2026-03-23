"""Tensor parallel weight splitting for Qwen2 models (Megatron-style)."""

import numpy as np


def split_column(tensor: np.ndarray, rank: int, world_size: int) -> np.ndarray:
    """Split tensor along dim 0 (output features). For Q/K/V/gate/up weights and biases."""
    chunk = tensor.shape[0] // world_size
    return tensor[rank * chunk : (rank + 1) * chunk].copy()


def split_row(tensor: np.ndarray, rank: int, world_size: int) -> np.ndarray:
    """Split tensor along dim 1 (input features). For attn_o/down weights."""
    chunk = tensor.shape[1] // world_size
    return tensor[:, rank * chunk : (rank + 1) * chunk].copy()


# Weight name patterns that get column-split (dim 0)
_COLUMN_SPLIT = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
}

# Weight name patterns that get row-split (dim 1)
_ROW_SPLIT = {
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
}


def shard_qwen2_weights(
    weights_dict: dict[str, np.ndarray], rank: int, world_size: int
) -> dict[str, np.ndarray]:
    """Shard Qwen2 model weights for tensor parallelism.

    Megatron-style: column split Q/K/V/gate/up, row split attn_o/down.
    Replicate: embeddings, norms, everything else.
    """
    if world_size <= 1:
        return weights_dict

    out = {}
    for name, tensor in weights_dict.items():
        # Extract the sub-key for layer weights (e.g. "self_attn.q_proj.weight")
        sub = None
        if name.startswith("model.layers."):
            parts = name.split(".")
            if len(parts) >= 4:
                sub = ".".join(parts[3:])

        if sub in _COLUMN_SPLIT:
            out[name] = split_column(tensor, rank, world_size)
        elif sub in _ROW_SPLIT:
            out[name] = split_row(tensor, rank, world_size)
        else:
            # Replicate: embeddings, norms, lm_head
            out[name] = tensor
    return out
