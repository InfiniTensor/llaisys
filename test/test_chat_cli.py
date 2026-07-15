import multiprocessing
import os
import socket
import subprocess
import sys


def _has_optional_deps() -> bool:
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        return False
    return True


if not _has_optional_deps():
    print("Skipped: optional server dependencies are not installed.")
    sys.exit(0)

from chat_test_utils import serve_fake_app, wait_for_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_cli_smoke() -> None:
    port = _free_port()
    process = multiprocessing.Process(target=serve_fake_app, args=(port,), daemon=True)
    process.start()
    try:
        wait_for_server(port)
        env = os.environ.copy()
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        python_paths = [
            os.path.join(repo_root, "python"),
            os.path.join(repo_root, "test"),
            env.get("PYTHONPATH", ""),
        ]
        env["PYTHONPATH"] = os.pathsep.join(path for path in python_paths if path)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llaisys.chat_cli",
                "--url",
                f"http://127.0.0.1:{port}",
                "--model",
                "fake-qwen",
                "--prompt",
                "hello",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Hello!" in result.stdout

        session_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llaisys.chat_cli",
                "--url",
                f"http://127.0.0.1:{port}",
                "--session-id",
                "cli-session",
                "--create-session",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert session_result.returncode == 0, session_result.stderr
        assert "cli-session" in session_result.stdout

        list_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llaisys.chat_cli",
                "--url",
                f"http://127.0.0.1:{port}",
                "--list-sessions",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert list_result.returncode == 0, list_result.stderr
        assert "cli-session" in list_result.stdout
    finally:
        process.terminate()
        process.join(timeout=5)


if __name__ == "__main__":
    test_cli_smoke()
    print("\033[92mTest passed!\033[0m\n")
