#pragma once
#include <string>
#include <vector>
#include <cstdio>
#include "document.hpp"
#include "core/hashing.hpp"
#include "core/config.hpp"

namespace slmkg {

namespace detail {

// Deterministic split assignment per spec Section 5.8:
// SHA256(document_id + str(seed)), first 4 bytes as uint32, mod 100.
// 0-79 -> train, 80-89 -> validation, 90-99 -> test.
inline std::string assign_split(const std::string& document_id, int seed,
                                 double train_frac, double val_frac) {
    std::string hex = sha256_hex(document_id + std::to_string(seed));
    uint32_t val = static_cast<uint32_t>(std::stoul(hex.substr(0, 8), nullptr, 16));
    int bucket = static_cast<int>(val % 100);
    int train_threshold = static_cast<int>(train_frac * 100);
    int val_threshold   = train_threshold + static_cast<int>(val_frac * 100);
    if (bucket < train_threshold) return "train";
    if (bucket < val_threshold)   return "validation";
    return "test";
}

inline std::string chunk_id(int n) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "chunk_%06d", n);
    return buf;
}

} // namespace detail

// Splits a DocumentRecord into ChunkRecords according to ChunkingConfig.
// next_chunk_id is the 1-based counter for the first chunk; it is advanced
// in-place so callers can assign globally unique chunk IDs across documents.
inline std::vector<ChunkRecord> chunk_document(const DocumentRecord& doc,
                                                const ChunkingConfig& cfg,
                                                const SplitsConfig&   splits_cfg,
                                                int& next_chunk_id) {
    std::vector<ChunkRecord> chunks;
    const std::string& text = doc.text;
    const int text_len = static_cast<int>(text.size());
    const int step = cfg.max_chunk_chars - cfg.overlap_chars;

    for (int start = 0; start < text_len; start += step) {
        int end = std::min(start + cfg.max_chunk_chars, text_len);
        int len = end - start;
        if (len < cfg.min_chunk_chars) break;  // too short — discard

        ChunkRecord c;
        c.id          = detail::chunk_id(next_chunk_id++);
        c.document_id = doc.id;
        c.chunk_index = static_cast<int>(chunks.size());
        c.text        = text.substr(start, len);
        c.char_start  = start;
        c.char_end    = end;
        c.token_count = len / 4;  // char_heuristic
        c.text_hash   = sha256_prefixed(c.text);
        c.split       = detail::assign_split(doc.id,
                                             splits_cfg.seed,
                                             splits_cfg.train,
                                             splits_cfg.validation);
        chunks.push_back(std::move(c));

        if (end == text_len) break;  // reached end of text
    }

    return chunks;
}

} // namespace slmkg
