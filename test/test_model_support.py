import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header = {}
    offset = 0
    raw_chunks = []
    for name in sorted(tensors):
        array = np.ascontiguousarray(tensors[name], dtype=np.float32)
        raw = array.tobytes(order="C")
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        raw_chunks.append(raw)
        offset += len(raw)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for chunk in raw_chunks:
            handle.write(chunk)


def _build_tiny_llama_dir(root: Path) -> Path:
    model_dir = root / "tiny-llama"
    model_dir.mkdir()

    config = {
        "model_type": "llama",
        "torch_dtype": "float32",
        "num_hidden_layers": 1,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 16,
        "max_position_embeddings": 32,
        "vocab_size": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
    }
    with open(model_dir / "config.json", "w") as handle:
        json.dump(config, handle)

    hs = config["hidden_size"]
    dh = hs // config["num_attention_heads"]
    nkvh = config["num_key_value_heads"]
    di = config["intermediate_size"]
    voc = config["vocab_size"]
    embed = np.arange(voc * hs, dtype=np.float32).reshape(voc, hs) * 0.01

    tensors = {
        "model.embed_tokens.weight": embed,
        "lm_head.weight": embed.copy(),
        "model.norm.weight": np.ones((hs,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.ones((hs,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.eye(hs, dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.eye(nkvh * dh, hs, dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.eye(nkvh * dh, hs, dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.eye(hs, dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.ones((hs,), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.ones((di, hs), dtype=np.float32) * 0.01,
        "model.layers.0.mlp.up_proj.weight": np.ones((di, hs), dtype=np.float32) * 0.01,
        "model.layers.0.mlp.down_proj.weight": np.ones((hs, di), dtype=np.float32) * 0.01,
    }
    _write_safetensors(model_dir / "model.safetensors", tensors)
    return model_dir


def _build_tiny_tied_llama_dir(root: Path) -> Path:
    model_dir = root / "tiny-llama-tied"
    model_dir.mkdir()

    config = {
        "model_type": "llama",
        "torch_dtype": "float32",
        "num_hidden_layers": 1,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 16,
        "max_position_embeddings": 32,
        "vocab_size": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "eos_token_id": 2,
        "tie_word_embeddings": True,
    }
    with open(model_dir / "config.json", "w") as handle:
        json.dump(config, handle)

    hs = config["hidden_size"]
    di = config["intermediate_size"]
    voc = config["vocab_size"]
    embed = np.arange(voc * hs, dtype=np.float32).reshape(voc, hs) * 0.01
    tensors = {
        "model.embed_tokens.weight": embed,
        "model.norm.weight": np.ones((hs,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.ones((hs,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.eye(hs, dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.eye(hs, dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.eye(hs, dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.eye(hs, dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.ones((hs,), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.ones((di, hs), dtype=np.float32) * 0.01,
        "model.layers.0.mlp.up_proj.weight": np.ones((di, hs), dtype=np.float32) * 0.01,
        "model.layers.0.mlp.down_proj.weight": np.ones((hs, di), dtype=np.float32) * 0.01,
    }
    _write_safetensors(model_dir / "model.safetensors", tensors)
    return model_dir


def test_llama_model_support() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = _build_tiny_llama_dir(Path(tmpdir))
        assert llaisys.models.detect_model_type(str(model_dir)) == "llama"
        assert llaisys.models.default_model_name(str(model_dir)) == "llaisys-llama"

        model = llaisys.models.create_model(str(model_dir), llaisys.DeviceType.CPU, 0)
        assert isinstance(model, llaisys.models.Llama)

        outputs = model.generate([1, 3, 5], max_new_tokens=1, top_k=1, top_p=1.0, temperature=1.0)
        assert outputs[:3] == [1, 3, 5]
        assert len(outputs) == 4

        tied_model_dir = _build_tiny_tied_llama_dir(Path(tmpdir))
        tied_model = llaisys.models.create_model(str(tied_model_dir), llaisys.DeviceType.CPU, 0)
        assert bool(tied_model.weights_ptr.contents.out_embed)


if __name__ == "__main__":
    test_llama_model_support()
    print("\033[92mTest passed!\033[0m\n")
