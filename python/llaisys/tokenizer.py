from __future__ import annotations

from ctypes import POINTER, c_char_p, c_int64, c_size_t, create_string_buffer
from pathlib import Path
from typing import Iterable, List, Optional

from .libllaisys import LIB_LLAISYS, LlaisysTokenizer


class Tokenizer:
    def __init__(self, model_path: str):
        self._backend: str = "sentencepiece"
        self._tokenizer: Optional[LlaisysTokenizer] = None
        self._hf_tokenizer = None

        tokenizer_path = self._resolve_tokenizer_path(model_path)
        if tokenizer_path.suffix.lower() == ".json":
            self._backend = "hf"
            self._hf_tokenizer = self._load_hf_tokenizer(tokenizer_path)
        else:
            self._tokenizer = LIB_LLAISYS.llaisysTokenizerCreateSentencePiece(
                c_char_p(str(tokenizer_path).encode("utf-8"))
            )
            if not self._tokenizer:
                raise RuntimeError("llaisysTokenizerCreateSentencePiece failed")

    def encode(self, text: str) -> List[int]:
        if self._backend == "hf":
            return list(self._hf_tokenizer.encode(text).ids)
        data = text.encode("utf-8")
        n = int(
            LIB_LLAISYS.llaisysTokenizerEncode(
                self._tokenizer, c_char_p(data), None, c_size_t(0)
            )
        )
        if n < 0:
            raise RuntimeError("llaisysTokenizerEncode failed")
        if n == 0:
            return []
        out_ids = (c_int64 * n)()
        written = int(
            LIB_LLAISYS.llaisysTokenizerEncode(
                self._tokenizer, c_char_p(data), out_ids, c_size_t(n)
            )
        )
        if written < 0:
            raise RuntimeError("llaisysTokenizerEncode failed")
        return [int(out_ids[i]) for i in range(written)]

    def decode(self, ids: Iterable[int]) -> str:
        ids_list = list(ids)
        n = len(ids_list)
        if n == 0:
            return ""
        if self._backend == "hf":
            return self._hf_tokenizer.decode(ids_list, skip_special_tokens=False)
        buf = (c_int64 * n)(*ids_list)
        max_len = int(
            LIB_LLAISYS.llaisysTokenizerDecode(
                self._tokenizer, buf, c_size_t(n), None, c_size_t(0)
            )
        )
        if max_len < 0:
            raise RuntimeError("llaisysTokenizerDecode failed")
        out = create_string_buffer(max_len)
        written = int(
            LIB_LLAISYS.llaisysTokenizerDecode(
                self._tokenizer, buf, c_size_t(n), out, c_size_t(max_len)
            )
        )
        if written < 0:
            raise RuntimeError("llaisysTokenizerDecode failed")
        return out.value.decode("utf-8")

    def close(self) -> None:
        if self._tokenizer:
            LIB_LLAISYS.llaisysTokenizerDestroy(self._tokenizer)
            self._tokenizer = None

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _resolve_tokenizer_path(model_path: str) -> Path:
        path = Path(model_path)
        if path.is_dir():
            sp = path / "tokenizer.model"
            if sp.exists():
                return sp
            hf = path / "tokenizer.json"
            if hf.exists():
                return hf
            raise FileNotFoundError(
                f"No tokenizer.model or tokenizer.json found under: {path}"
            )
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        return path

    @staticmethod
    def _load_hf_tokenizer(path: Path):
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except Exception as exc:
            raise RuntimeError(
                "tokenizer.json requires the 'tokenizers' package. "
                "Install with: pip install tokenizers"
            ) from exc
        return HFTokenizer.from_file(str(path))


__all__ = ["Tokenizer"]
