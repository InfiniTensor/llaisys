import os
import sys


def setup_paths(current_file: str) -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(current_file), ".."))
    python_dir = os.path.join(repo_root, "python")
    test_dir = os.path.join(repo_root, "test")

    for path in (python_dir, test_dir, repo_root):
        if path not in sys.path:
            sys.path.insert(0, path)
