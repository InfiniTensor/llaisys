import uvicorn

from llaisys.chat.server import build_runtime_from_env, create_app


def main() -> None:
    runtime = build_runtime_from_env()
    app = create_app(runtime)
    uvicorn.run(app, host="0.0.0.0", port=9108)


if __name__ == "__main__":
    main()
