import importlib
import os
import sys
import tempfile


def _load_server_config_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_root = os.path.join(repo_root, "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)

    if "server.config" in sys.modules:
        del sys.modules["server.config"]

    return importlib.import_module("server.config")


def _set_model_path_env(value, monkeypatch=None):
    env_key = "LLAISYS_MODEL_PATH"
    if monkeypatch is not None:
        monkeypatch.setenv(env_key, value)
        return None

    previous = os.environ.get(env_key)
    os.environ[env_key] = value
    return previous


def _restore_model_path_env(previous, monkeypatch=None):
    env_key = "LLAISYS_MODEL_PATH"
    if monkeypatch is not None:
        return
    if previous is None:
        os.environ.pop(env_key, None)
    else:
        os.environ[env_key] = previous


def test_model_path_uses_existing_local_path(monkeypatch=None):
    with tempfile.TemporaryDirectory() as temp_dir:
        previous = _set_model_path_env(temp_dir, monkeypatch)
        try:
            config_module = _load_server_config_module()
            config = config_module.ServerConfig()
            assert config.model_path == temp_dir
        finally:
            _restore_model_path_env(previous, monkeypatch)


def test_model_path_fallbacks_to_hf_when_path_missing(monkeypatch=None):
    previous = _set_model_path_env("/tmp/llaisys-model-path-not-exist", monkeypatch)
    try:
        config_module = _load_server_config_module()
        config = config_module.ServerConfig()
        assert config.model_path is None
    finally:
        _restore_model_path_env(previous, monkeypatch)


if __name__ == "__main__":
    test_model_path_uses_existing_local_path()
    test_model_path_fallbacks_to_hf_when_path_missing()
    print("\033[92mTest passed!\033[0m\n")
