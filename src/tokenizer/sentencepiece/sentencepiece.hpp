#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llaisys::tokenizer {

class SentencePieceTokenizer {
public:
    explicit SentencePieceTokenizer(const std::string &model_path);
    ~SentencePieceTokenizer();

    bool isLoaded() const;

    bool encode(const std::string &text, std::vector<int64_t> &out_ids) const;
    bool decode(const int64_t *ids, size_t len, std::string &out_text) const;

private:
#ifdef LLAISYS_ENABLE_SENTENCEPIECE
    class Impl;
    Impl *_impl{nullptr};
#endif
};

} // namespace llaisys::tokenizer
