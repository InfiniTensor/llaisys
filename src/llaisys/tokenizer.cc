#include "llaisys/tokenizer.h"

#include "../tokenizer/sentencepiece/sentencepiece.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct LlaisysTokenizer {
    std::unique_ptr<llaisys::tokenizer::SentencePieceTokenizer> impl;
};

__C {
__export struct LlaisysTokenizer *llaisysTokenizerCreateSentencePiece(const char *model_path) {
    if (!model_path || model_path[0] == '\0') return nullptr;
    auto tokenizer = std::make_unique<LlaisysTokenizer>();
    tokenizer->impl = std::make_unique<llaisys::tokenizer::SentencePieceTokenizer>(model_path);
    if (!tokenizer->impl || !tokenizer->impl->isLoaded()) {
        return nullptr;
    }
    return tokenizer.release();
}

__export void llaisysTokenizerDestroy(struct LlaisysTokenizer *tokenizer) {
    delete tokenizer;
}

__export int llaisysTokenizerEncode(struct LlaisysTokenizer *tokenizer,
                                    const char *text,
                                    int64_t *out_ids,
                                    size_t max_ids) {
    if (!tokenizer || !tokenizer->impl || !text) return -1;
    std::vector<int64_t> ids;
    if (!tokenizer->impl->encode(text, ids)) return -1;
    if (!out_ids || max_ids == 0) {
        return static_cast<int>(ids.size());
    }
    const size_t n = ids.size() < max_ids ? ids.size() : max_ids;
    for (size_t i = 0; i < n; ++i) out_ids[i] = ids[i];
    return static_cast<int>(n);
}

__export int llaisysTokenizerDecode(struct LlaisysTokenizer *tokenizer,
                                    const int64_t *ids,
                                    size_t len,
                                    char *out_text,
                                    size_t max_len) {
    if (!tokenizer || !tokenizer->impl) return -1;
    std::string text;
    if (!tokenizer->impl->decode(ids, len, text)) return -1;
    if (!out_text || max_len == 0) {
        return static_cast<int>(text.size() + 1);
    }
    const size_t n = text.size() < (max_len - 1) ? text.size() : (max_len - 1);
    std::memcpy(out_text, text.data(), n);
    out_text[n] = '\0';
    return static_cast<int>(n);
}
}
