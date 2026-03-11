import os
import subprocess
import sys

from bootstrap import setup_paths

setup_paths(__file__)

TEST_ROOT = os.path.abspath(os.path.dirname(__file__))
TEST_OPS_DIR = os.path.join(TEST_ROOT, "ops")


def run_tests(args):
    failed = []
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                os.path.join(os.path.dirname(TEST_ROOT), "python"),
                TEST_ROOT,
                env.get("PYTHONPATH", ""),
            ],
        )
    )
    for test in [
        "add.py",
        "argmax.py",
        "embedding.py",
        "linear.py",
        "rms_norm.py",
        "rope.py",
        "self_attention.py",
        "swiglu.py",
    ]:
        result = subprocess.run(
            [sys.executable, os.path.join(TEST_OPS_DIR, test), *sys.argv[1:]],
            text=True,
            encoding="utf-8",
            env=env,
        )
        if result.returncode != 0:
            failed.append(test)

    return failed


if __name__ == "__main__":
    failed = run_tests(" ".join(sys.argv[1:]))
    if len(failed) == 0:
        print("\033[92mAll tests passed!\033[0m")
    else:
        print("\033[91mThe following tests failed:\033[0m")
        for test in failed:
            print(f"\033[91m - {test}\033[0m")
    exit(len(failed))
