#pragma once
#include <string>
#include <nlohmann/json.hpp>

namespace slmkg {

struct DocumentRecord {
    std::string id;
    std::string source_path;
    std::string source_type;   // "markdown", "text", "jsonl"
    std::string title;
    std::string author;
    std::string created_at;
    std::string ingested_at;
    std::string text_hash;
    std::string language     = "en";
    std::string license;
    std::string trust_level  = "trusted";
    std::string domain;
    std::string split;         // "train", "validation", "test"
    // Internal — not written to documents.jsonl
    std::string text;
};

inline void to_json(nlohmann::json& j, const DocumentRecord& d) {
    j = {
        {"id",           d.id},
        {"source_path",  d.source_path},
        {"source_type",  d.source_type},
        {"title",        d.title},
        {"ingested_at",  d.ingested_at},
        {"text_hash",    d.text_hash},
        {"language",     d.language},
        {"trust_level",  d.trust_level},
    };
    if (!d.author.empty())     j["author"]     = d.author;
    if (!d.created_at.empty()) j["created_at"] = d.created_at;
    if (!d.license.empty())    j["license"]    = d.license;
    if (!d.domain.empty())     j["domain"]     = d.domain;
    if (!d.split.empty())      j["split"]      = d.split;
}

// -----------------------------------------------------------------------

struct ChunkRecord {
    std::string id;
    std::string document_id;
    int         chunk_index  = 0;
    std::string text;
    int         char_start   = 0;
    int         char_end     = 0;
    int         token_count  = 0;
    std::string text_hash;
    std::string split;
};

inline void to_json(nlohmann::json& j, const ChunkRecord& c) {
    j = {
        {"id",           c.id},
        {"document_id",  c.document_id},
        {"chunk_index",  c.chunk_index},
        {"text",         c.text},
        {"char_start",   c.char_start},
        {"char_end",     c.char_end},
        {"token_count",  c.token_count},
        {"text_hash",    c.text_hash},
    };
    if (!c.split.empty()) j["split"] = c.split;
}

} // namespace slmkg
