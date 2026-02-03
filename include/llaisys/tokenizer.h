#ifndef LLAISYS_TOKENIZER_H
#define LLAISYS_TOKENIZER_H

#include "../llaisys.h"

__C {
    struct LlaisysTokenizer;

    // Create a SentencePiece tokenizer from model file path.
    __export struct LlaisysTokenizer *llaisysTokenizerCreateSentencePiece(const char *model_path);

    // Destroy tokenizer instance.
    __export void llaisysTokenizerDestroy(struct LlaisysTokenizer *tokenizer);

    // Encode text into token ids.
    // If out_ids is null or max_ids is 0, returns the required length.
    // On error returns -1.
    __export int llaisysTokenizerEncode(struct LlaisysTokenizer *tokenizer,
                                        const char *text,
                                        int64_t *out_ids,
                                        size_t max_ids);

    // Decode token ids into text.
    // If out_text is null or max_len is 0, returns the required length (including null terminator).
    // On error returns -1.
    __export int llaisysTokenizerDecode(struct LlaisysTokenizer *tokenizer,
                                        const int64_t *ids,
                                        size_t len,
                                        char *out_text,
                                        size_t max_len);
}

#endif // LLAISYS_TOKENIZER_H
