from ctypes import POINTER, c_char_p, c_int, c_int64, c_size_t, c_void_p


LlaisysTokenizer = c_void_p


def load_tokenizer(lib):
    lib.llaisysTokenizerCreateSentencePiece.argtypes = [c_char_p]
    lib.llaisysTokenizerCreateSentencePiece.restype = LlaisysTokenizer

    lib.llaisysTokenizerDestroy.argtypes = [LlaisysTokenizer]
    lib.llaisysTokenizerDestroy.restype = None

    lib.llaisysTokenizerEncode.argtypes = [
        LlaisysTokenizer,
        c_char_p,
        POINTER(c_int64),
        c_size_t,
    ]
    lib.llaisysTokenizerEncode.restype = c_int

    lib.llaisysTokenizerDecode.argtypes = [
        LlaisysTokenizer,
        POINTER(c_int64),
        c_size_t,
        c_char_p,
        c_size_t,
    ]
    lib.llaisysTokenizerDecode.restype = c_int


__all__ = ["LlaisysTokenizer", "load_tokenizer"]
