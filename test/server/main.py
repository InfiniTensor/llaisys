"""
main.py — FastAPI application entry point

This file sets up the FastAPI app, mounts static files, and provides the
CLI entry point for starting the chatbot server.

Usage:
    python -m llaisys.server.main --model /path/to/model --port 8000

Architecture overview:
    Browser  <──SSE──>  FastAPI (main.py + routes.py)  <──ctypes──>  C++ backend
                              │
                         Tokenizer (HuggingFace transformers)

The server is intentionally single-user and single-threaded on the model side:
one request is served at a time. This matches the Project #3 requirement and
keeps the code simple. Multi-user batching is covered in Project #4.
"""

import argparse
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from routes import router, set_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI application.
# Including the router from routes.py registers all API endpoints.
app = FastAPI(title="LLAISYS Chatbot")
app.include_router(router)

# Serve static files (HTML, CSS, JS) under the /static URL prefix.
# Path(__file__).parent resolves to the directory containing this file,
# so "static/" is always found relative to main.py regardless of where
# the server is launched from.
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    """Serve the chat UI. The browser loads this on first visit."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Returns whether the model has been loaded successfully.
    Useful for monitoring or waiting for the server to be ready.
    """
    from .routes import MODEL
    return {"status": "ok", "model_loaded": MODEL is not None}


def main():
    """
    CLI entry point: load the model then start the HTTP server.

    Model loading happens before uvicorn starts so that the first request
    doesn't have to wait. The model stays in memory for the lifetime of
    the server process.
    """
    parser = argparse.ArgumentParser(description="LLAISYS Chatbot Server")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    import llaisys

    logger.info(f"Loading model from {args.model} ...")
    device = llaisys.DeviceType.CPU if args.device == "cpu" else llaisys.DeviceType.NVIDIA

    # Load the LLAISYS C++ model (weights are read from safetensors files)
    model = llaisys.models.Qwen2(args.model, device=device)

    # Load the tokenizer from HuggingFace — used to encode prompts and
    # decode generated token IDs back to text
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Make model and tokenizer available to the route handlers
    set_model(model, tokenizer)
    logger.info(f"Model loaded. Starting server at http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
