#pragma once
#include <string>
#include <vector>
#include "errors.hpp"
#include <yaml-cpp/yaml.h>

namespace slmkg {

struct ChunkingConfig {
    int         min_chunk_chars    = 300;
    int         max_chunk_chars    = 2000;
    int         overlap_chars      = 100;
    std::string token_count_method = "char_heuristic";
};

struct DeduplicationConfig {
    bool        enabled = true;
    std::string method  = "exact_hash";
    std::string scope   = "chunks";
};

struct SplitsConfig {
    double      train      = 0.80;
    double      validation = 0.10;
    double      test       = 0.10;
    int         seed       = 42;
    std::string split_by   = "document_id";
};

struct QualityConfig {
    double                   min_language_score     = 0.80;
    std::vector<std::string> trusted_source_types   = {"txt", "md", "jsonl", "pdf"};
    bool                     warn_missing_author    = true;
    bool                     warn_missing_license   = true;
    bool                     fail_on_duplicate_ids  = true;
    bool                     fail_on_malformed_jsonl= true;
    bool                     fail_on_unreadable_files = true;
};

struct ManifestConfig {
    bool        write          = true;
    std::string hash_algorithm = "sha256";
};

struct CorpusConfig {
    std::string        id;
    std::string        source_root;
    std::string        output_root;
    std::string        encoding = "utf-8";
    ChunkingConfig     chunking;
    DeduplicationConfig deduplication;
    SplitsConfig       splits;
    QualityConfig      quality;
    ManifestConfig     manifest;
};

inline Result<CorpusConfig> load_corpus_config(const std::string& path) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        YAML::Node c = root["corpus"];
        if (!c) return Result<CorpusConfig>::err("Missing 'corpus' key in " + path);

        CorpusConfig cfg;
        cfg.id          = c["id"]          ? c["id"].as<std::string>()          : "";
        cfg.source_root = c["source_root"] ? c["source_root"].as<std::string>() : "";
        cfg.output_root = c["output_root"] ? c["output_root"].as<std::string>() : "";
        cfg.encoding    = c["encoding"]    ? c["encoding"].as<std::string>()    : "utf-8";

        if (YAML::Node ch = c["chunking"]) {
            if (ch["min_chunk_chars"])    cfg.chunking.min_chunk_chars    = ch["min_chunk_chars"].as<int>();
            if (ch["max_chunk_chars"])    cfg.chunking.max_chunk_chars    = ch["max_chunk_chars"].as<int>();
            if (ch["overlap_chars"])      cfg.chunking.overlap_chars      = ch["overlap_chars"].as<int>();
            if (ch["token_count_method"]) cfg.chunking.token_count_method = ch["token_count_method"].as<std::string>();
        }

        if (YAML::Node d = c["deduplication"]) {
            if (d["enabled"]) cfg.deduplication.enabled = d["enabled"].as<bool>();
            if (d["method"])  cfg.deduplication.method  = d["method"].as<std::string>();
            if (d["scope"])   cfg.deduplication.scope   = d["scope"].as<std::string>();
        }

        if (YAML::Node s = c["splits"]) {
            if (s["train"])      cfg.splits.train      = s["train"].as<double>();
            if (s["validation"]) cfg.splits.validation = s["validation"].as<double>();
            if (s["test"])       cfg.splits.test       = s["test"].as<double>();
            if (s["seed"])       cfg.splits.seed       = s["seed"].as<int>();
            if (s["split_by"])   cfg.splits.split_by   = s["split_by"].as<std::string>();
        }

        if (YAML::Node q = c["quality"]) {
            if (q["min_language_score"])      cfg.quality.min_language_score      = q["min_language_score"].as<double>();
            if (q["warn_missing_author"])     cfg.quality.warn_missing_author     = q["warn_missing_author"].as<bool>();
            if (q["warn_missing_license"])    cfg.quality.warn_missing_license    = q["warn_missing_license"].as<bool>();
            if (q["fail_on_duplicate_ids"])   cfg.quality.fail_on_duplicate_ids   = q["fail_on_duplicate_ids"].as<bool>();
            if (q["fail_on_malformed_jsonl"]) cfg.quality.fail_on_malformed_jsonl = q["fail_on_malformed_jsonl"].as<bool>();
            if (q["fail_on_unreadable_files"])cfg.quality.fail_on_unreadable_files= q["fail_on_unreadable_files"].as<bool>();
            if (q["trusted_source_types"]) {
                cfg.quality.trusted_source_types.clear();
                for (const auto& t : q["trusted_source_types"])
                    cfg.quality.trusted_source_types.push_back(t.as<std::string>());
            }
        }

        if (YAML::Node m = c["manifest"]) {
            if (m["write"])          cfg.manifest.write          = m["write"].as<bool>();
            if (m["hash_algorithm"]) cfg.manifest.hash_algorithm = m["hash_algorithm"].as<std::string>();
        }

        return Result<CorpusConfig>::ok(std::move(cfg));
    } catch (const YAML::Exception& e) {
        return Result<CorpusConfig>::err(std::string("YAML error loading ") + path + ": " + e.what());
    }
}

} // namespace slmkg
