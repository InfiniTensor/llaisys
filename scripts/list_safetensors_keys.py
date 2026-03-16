"""List keys in .safetensors files without loading tensor data (metadata only)."""
import sys
from pathlib import Path

import safetensors


def main():
    if len(sys.argv) < 2:
        print("Usage: python list_safetensors_keys.py <model_dir>")
        sys.exit(1)
    model_dir = Path(sys.argv[1])
    if not model_dir.is_dir():
        print("Not a directory:", model_dir)
        sys.exit(1)
    for fpath in sorted(model_dir.glob("*.safetensors")):
        print("\n---", fpath.name, "---")
        with safetensors.safe_open(fpath, framework="numpy", device="cpu") as f:
            keys = list(f.keys())
        for k in keys:
            print(k)
        print("Total keys:", len(keys))


if __name__ == "__main__":
    main()
