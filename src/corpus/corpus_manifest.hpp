#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "document.hpp"
#include "core/config.hpp"
#include "core/hashing.hpp"
#include "core/errors.hpp"
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

namespace slmkg {

struct CorpusStats {
    int    document_count    = 0;
    int    chunk_count       = 0;
    double avg_chunk_chars   = 0.0;
    int    train_count       = 0;
    int    validation_count  = 0;
    int    test_count        = 0;
};

inline CorpusStats compute_stats(const std::vector<DocumentRecord>& docs,
                                  const std::vector<ChunkRecord>&    chunks) {
    CorpusStats s;
    s.document_count = static_cast<int>(docs.size());
    s.chunk_count    = static_cast<int>(chunks.size());

    long long total_chars = 0;
    for (const auto& c : chunks) {
        total_chars += c.char_end - c.char_start;
        if      (c.split == "train")      ++s.train_count;
        else if (c.split == "validation") ++s.validation_count;
        else if (c.split == "test")       ++s.test_count;
    }
    if (s.chunk_count > 0)
        s.avg_chunk_chars = static_cast<double>(total_chars) / s.chunk_count;

    return s;
}

// Writes documents.jsonl and chunks.jsonl to output_dir.
// Returns the SHA256 of the full chunks.jsonl content.
inline Result<std::string> write_corpus_jsonl(
        const std::string&                 output_dir,
        const std::vector<DocumentRecord>& docs,
        const std::vector<ChunkRecord>&    chunks) {
    namespace fs = std::filesystem;
    try {
        fs::create_directories(output_dir);

        std::ostringstream chunks_content;

        {
            std::ofstream f(output_dir + "/documents.jsonl");
            for (const auto& d : docs) {
                nlohmann::json j = d;
                f << j.dump() << "\n";
            }
        }
        {
            std::ofstream f(output_dir + "/chunks.jsonl");
            for (const auto& c : chunks) {
                nlohmann::json j = c;
                std::string line = j.dump();
                f << line << "\n";
                chunks_content << line << "\n";
            }
        }

        return Result<std::string>::ok(sha256_prefixed(chunks_content.str()));
    } catch (const std::exception& e) {
        return Result<std::string>::err(std::string("Write error: ") + e.what());
    }
}

// Writes corpus_manifest.yaml to output_dir.
inline Result<void> write_manifest(const std::string&  output_dir,
                                    const CorpusConfig& cfg,
                                    const CorpusStats&  stats,
                                    const std::string&  chunks_hash,
                                    const std::string&  created_at) {
    try {
        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "corpus" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "id"             << YAML::Value << cfg.id;
        out << YAML::Key << "created_at"     << YAML::Value << created_at;
        out << YAML::Key << "source_root"    << YAML::Value << cfg.source_root;
        out << YAML::Key << "document_count" << YAML::Value << stats.document_count;
        out << YAML::Key << "chunk_count"    << YAML::Value << stats.chunk_count;
        out << YAML::Key << "min_chunk_chars"<< YAML::Value << cfg.chunking.min_chunk_chars;
        out << YAML::Key << "max_chunk_chars"<< YAML::Value << cfg.chunking.max_chunk_chars;
        out << YAML::Key << "deduplication"  << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "enabled"        << YAML::Value << cfg.deduplication.enabled;
        out << YAML::Key << "method"         << YAML::Value << cfg.deduplication.method;
        out << YAML::EndMap;
        out << YAML::Key << "splits" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "train"      << YAML::Value << cfg.splits.train;
        out << YAML::Key << "validation" << YAML::Value << cfg.splits.validation;
        out << YAML::Key << "test"       << YAML::Value << cfg.splits.test;
        out << YAML::EndMap;
        out << YAML::Key << "hash" << YAML::Value << chunks_hash;
        out << YAML::EndMap;
        out << YAML::EndMap;

        std::ofstream f(output_dir + "/corpus_manifest.yaml");
        f << out.c_str() << "\n";
        return Result<void>::ok();
    } catch (const std::exception& e) {
        return Result<void>::err(std::string("Manifest write error: ") + e.what());
    }
}

// Writes corpus_stats.json to output_dir.
inline Result<void> write_stats_json(const std::string& output_dir,
                                      const CorpusStats& stats) {
    try {
        nlohmann::json j = {
            {"document_count",   stats.document_count},
            {"chunk_count",      stats.chunk_count},
            {"avg_chunk_chars",  stats.avg_chunk_chars},
            {"train_count",      stats.train_count},
            {"validation_count", stats.validation_count},
            {"test_count",       stats.test_count},
        };
        std::ofstream f(output_dir + "/corpus_stats.json");
        f << j.dump(2) << "\n";
        return Result<void>::ok();
    } catch (const std::exception& e) {
        return Result<void>::err(std::string("Stats write error: ") + e.what());
    }
}

} // namespace slmkg
