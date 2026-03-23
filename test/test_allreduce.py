"""Multi-process allreduce integration test.

Launches N worker processes (one per GPU), each initializing a NCCL communicator
and performing allreduce. Uses file-based IPC to broadcast the NCCL unique ID
from rank 0 to all other ranks.

Usage:
    python test_allreduce.py [--nranks 2] [--device nvidia]
"""

import sys
import os
import subprocess
import argparse
import tempfile
import struct

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)


WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_allreduce_worker.py")


def run_allreduce_test(nranks, device):
    """Launch nranks worker processes and verify allreduce results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        id_file = os.path.join(tmpdir, "nccl_id.bin")
        result_files = [os.path.join(tmpdir, f"result_{r}.bin") for r in range(nranks)]

        procs = []
        for rank in range(nranks):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(rank)
            proc = subprocess.Popen(
                [
                    sys.executable, WORKER_SCRIPT,
                    "--rank", str(rank),
                    "--nranks", str(nranks),
                    "--device", device,
                    "--id_file", id_file,
                    "--result_file", result_files[rank],
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            procs.append(proc)

        # Wait for all workers
        failed = False
        for rank, proc in enumerate(procs):
            stdout, stderr = proc.communicate(timeout=60)
            if proc.returncode != 0:
                print(f"Rank {rank} FAILED (exit code {proc.returncode}):")
                print(stderr.decode(errors="replace"))
                failed = True

        if failed:
            raise RuntimeError("One or more workers failed")

        # Verify results: each rank sends [rank+1]*4, allreduce SUM => [sum(1..N)]*4
        expected_val = sum(r + 1.0 for r in range(nranks))
        for rank in range(nranks):
            with open(result_files[rank], "rb") as f:
                data = f.read()
            result = struct.unpack("ffff", data)
            for i, v in enumerate(result):
                assert abs(v - expected_val) < 1e-3, (
                    f"Rank {rank} result[{i}] = {v}, expected {expected_val}"
                )

        print(f"Allreduce SUM verified: all {nranks} ranks produced {expected_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nranks", type=int, default=2)
    parser.add_argument("--device", default="nvidia", choices=["nvidia", "iluvatar"])
    args = parser.parse_args()

    print(f"=== Multi-process allreduce test ({args.nranks} ranks) ===")
    run_allreduce_test(args.nranks, args.device)
    print("\n\033[92mAllreduce integration test passed!\033[0m\n")
