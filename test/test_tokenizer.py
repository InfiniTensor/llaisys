import argparse
import os
from ctypes import c_char_p, c_int64, c_size_t, create_string_buffer

from llaisys.libllaisys import LIB_LLAISYS


def test_sentencepiece(model_path: str, text: str):
    tokenizer = LIB_LLAISYS.llaisysTokenizerCreateSentencePiece(model_path.encode("utf-8"))
    if not tokenizer:
        print("SentencePiece tokenizer not available or model load failed. Skipped.")
        return

    # query required length
    needed = LIB_LLAISYS.llaisysTokenizerEncode(tokenizer, text.encode("utf-8"), None, c_size_t(0))
    assert needed > 0

    ids = (c_int64 * needed)()
    n = LIB_LLAISYS.llaisysTokenizerEncode(tokenizer, text.encode("utf-8"), ids, c_size_t(needed))
    assert n > 0

    # query decode length
    decode_needed = LIB_LLAISYS.llaisysTokenizerDecode(tokenizer, ids, c_size_t(n), None, c_size_t(0))
    assert decode_needed > 0

    out = create_string_buffer(decode_needed)
    nbytes = LIB_LLAISYS.llaisysTokenizerDecode(tokenizer, ids, c_size_t(n), out, c_size_t(decode_needed))
    assert nbytes >= 0
    decoded = out.value.decode("utf-8")
    assert decoded != ""

    LIB_LLAISYS.llaisysTokenizerDestroy(tokenizer)
    print("Encoded ids:", list(ids)[: min(8, n)], "...")
    print("Decoded text:", decoded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LLAISYS_TOKENIZER_MODEL", ""), type=str)
    parser.add_argument("--text", default="我喜欢人工智能", type=str)
    args = parser.parse_args()

    if not args.model:
        print("No SentencePiece model path provided. Set --model or LLAISYS_TOKENIZER_MODEL. Skipped.")
    else:
        test_sentencepiece(args.model, args.text)
        print("\033[92mTest passed!\033[0m\n")
