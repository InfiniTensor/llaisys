#!/usr/bin/env python3
"""Tensor-parallel multi-process launcher for llaisys inference.

Rank 0 generates a NCCL unique ID, writes it to a temp file, then spawns
N subprocesses (one per GPU). Each subprocess loads sharded weights, inits
the communicator with the shared unique ID, and runs inference. Output is
printed from rank 0.

Usage:
    python scripts/launch_tp.py --model /path/to/qwen2 --nranks 2 --prompt "Hello"
"""

import argparse
import os
import sys
import subprocess
import tempfile
import ctypes

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)


def generate_unique_id(backend=0):
    """Generate NCCL unique ID via llaisysCommGenerateUniqueId."""
    from llaisys.libllaisys import LIB_LLAISYS, LLAISYS_COMM_UNIQUE_ID_MAX_SIZE
    id_buf = ctypes.create_string_buffer(LLAISYS_COMM_UNIQUE_ID_MAX_SIZE)
    id_size = ctypes.c_size_t(0)
    ret = LIB_LLAISYS.llaisysCommGenerateUniqueId(
        backend, id_buf, ctypes.byref(id_size)
    )
    if ret != 0:
        raise RuntimeError(f"llaisysCommGenerateUniqueId failed: {ret}")
    return id_buf.raw[: id_size.value]


def main():
    parser = argparse.ArgumentParser(description="Tensor-parallel launcher")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--nranks", type=int, default=2, help="Number of TP ranks")
    parser.add_argument("--device", default="nvidia", choices=["nvidia", "iluvatar"])
    parser.add_argument("--prompt", default="Hello", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    # Generate unique ID on rank 0 process
    uid_bytes = generate_unique_id()

    # Write unique ID to temp file for subprocesses
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".uid")
    tmp.write(uid_bytes)
    tmp.close()
    uid_path = tmp.name

    worker = os.path.join(os.path.dirname(__file__), "_tp_worker.py")

    procs = []
    for rank in range(args.nranks):
        env = os.environ.copy()
        env["RANK"] = str(rank)
        env["WORLD_SIZE"] = str(args.nranks)
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        env["TP_UID_FILE"] = uid_path
        env["TP_MODEL_PATH"] = args.model
        env["TP_DEVICE"] = args.device
        env["TP_PROMPT"] = args.prompt
        env["TP_MAX_TOKENS"] = str(args.max_tokens)

        proc = subprocess.Popen(
            [sys.executable, worker],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append((rank, proc))

    # Wait for all and collect output
    for rank, proc in procs:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(f"[rank {rank}] FAILED (exit {proc.returncode})", file=sys.stderr)
            if stderr:
                print(stderr.decode(errors="replace"), file=sys.stderr)
        elif rank == 0:
            print(stdout.decode(errors="replace"), end="")

    # Cleanup
    try:
        os.unlink(uid_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
