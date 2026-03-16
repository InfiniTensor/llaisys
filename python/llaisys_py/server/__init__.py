"""
Chatbot HTTP server (OpenAI chat-completion style).
Run with: python -m llaisys_py.server
"""

from .app import create_app

__all__ = ["create_app"]
