import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia", "iluvatar"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    scripts = [
        "add.py",
        "argmax.py",
        "embedding.py",
        "linear.py",
        "rearrange.py",
        "rms_norm.py",
        "rope.py",
        "self_attention.py",
        "swiglu.py",
    ]

    print(f"Running GPU op tests on {args.device}")
    for name in scripts:
        cmd = [sys.executable, str(here / name), "--device", args.device]
        if args.profile:
            cmd.append("--profile")
        print(f"\n=== {name} ===")
        result = subprocess.run(cmd, cwd=str(here))
        if result.returncode != 0:
            print(f"[ERROR] {name} failed with code {result.returncode}")
            return result.returncode

    print("\nAll GPU op tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
