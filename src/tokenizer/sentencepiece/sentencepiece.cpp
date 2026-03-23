#include "sentencepiece.hpp"

#include <iostream>

#ifdef LLAISYS_ENABLE_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

namespace llaisys::tokenizer {

#ifdef LLAISYS_ENABLE_SENTENCEPIECE
class SentencePieceTokenizer::Impl {
public:
    bool load(const std::string &model_path) {
        auto status = _sp.Load(model_path);
        return status.ok();
    }

    bool encode(const std::string &text, std::vector<int64_t> &out_ids) const {
        std::vector<int> ids;
        auto status = _sp.Encode(text, &ids);
        if (!status.ok()) return false;
        out_ids.assign(ids.begin(), ids.end());
        return true;
    }

    bool decode(const int64_t *ids, size_t len, std::string &out_text) const {
        if (!ids && len > 0) return false;
        std::vector<int> tmp;
        tmp.reserve(len);
        for (size_t i = 0; i < len; ++i) tmp.push_back(static_cast<int>(ids[i]));
        auto status = _sp.Decode(tmp, &out_text);
        return status.ok();
    }

    bool pieceToId(const std::string &piece, int64_t &out_id) const {
        int id = _sp.PieceToId(piece);
        if (id < 0) return false;
        std::string check = _sp.IdToPiece(id);
        if (check != piece) return false;
        out_id = static_cast<int64_t>(id);
        return true;
    }

    bool idToPiece(int64_t id, std::string &out_piece) const {
        if (id < 0) return false;
        if (id >= static_cast<int64_t>(_sp.GetPieceSize())) return false;
        out_piece = _sp.IdToPiece(static_cast<int>(id));
        return !out_piece.empty();
    }

private:
    sentencepiece::SentencePieceProcessor _sp;
};
#endif

SentencePieceTokenizer::SentencePieceTokenizer(const std::string &model_path) {
#ifdef LLAISYS_ENABLE_SENTENCEPIECE
    _impl = new Impl();
    if (!_impl->load(model_path)) {
        std::cerr << "[ERROR] SentencePiece load failed: " << model_path << std::endl;
        delete _impl;
        _impl = nullptr;
    }
#else
    (void)model_path;
    std::cerr << "[ERROR] SentencePiece is not enabled in build." << std::endl;
#endif
}

SentencePieceTokenizer::~SentencePieceTokenizer() {
#ifdef LLAISYS_ENABLE_SENTENCEPIECE
    delete _impl;
    _impl = nullptr;
#endif
}

bool SentencePieceTokenizer::isLoaded() const {
#ifdef LLAISYS_ENABLE_SENTENCEPIECE
    return _impl != nullptr;
#else
    return false;
#endif
}

bool SentencePieceTokenizer::encode(const std::string &text, std::vector<int64_t> &out_ids) const {
#ifdef LLAISYS_ENABLE_SENTENCEPIECE
    if (!_impl) return false;
    return _impl->encode(text, out_ids);
#else
    (void)text;
    out_ids.clear();
    return false;
#endif
}

bool SentencePieceTokenizer::decode(const int64_t *ids, size_t len, std::string &out_text) const {
#ifdef LLAISYS_ENABLE_SENTENCEPIECE
    if (!_impl) return false;
    return _impl->decode(ids, len, out_text);
#else
    (void)ids;
    (void)len;
    out_text.clear();
    return false;
#endif
}

bool SentencePieceTokenizer::pieceToId(const std::string &piece, int64_t &out_id) const {
#ifdef LLAISYS_ENABLE_SENTENCEPIECE
    if (!_impl) return false;
    return _impl->pieceToId(piece, out_id);
#else
    (void)piece;
    out_id = -1;
    return false;
#endif
}

bool SentencePieceTokenizer::idToPiece(int64_t id, std::string &out_piece) const {
#ifdef LLAISYS_ENABLE_SENTENCEPIECE
    if (!_impl) return false;
    return _impl->idToPiece(id, out_piece);
#else
    (void)id;
    out_piece.clear();
    return false;
#endif
}

} // namespace llaisys::tokenizer
