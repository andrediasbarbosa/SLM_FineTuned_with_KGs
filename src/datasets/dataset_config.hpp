#pragma once
#include <string>
#include <random>
#include "core/errors.hpp"
#include <yaml-cpp/yaml.h>

namespace slmkg {

struct DatasetConfig {
    std::string kg_snapshot_id      = "";
    std::string corpus_chunks_path  = ""; // optional: path to chunks.jsonl for raw_text records
    int         max_records         = 10000;
    int         max_char_length     = 3600;   // T4.6 length filter (~900 tokens)
    int         split_seed          = 42;
    double      train_ratio         = 0.80;
    double      val_ratio           = 0.10;
    double      test_ratio          = 0.10;
    int         max_hop_paths       = 500;    // cap on 2-hop path enumeration
    double      refusal_ratio       = 0.15;  // fraction of SFT records that are refusals
    std::string generator           = "builder_v0.1";
};

inline Result<DatasetConfig> load_dataset_config(const std::string& path) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        YAML::Node c = root["dataset"];
        if (!c) return Result<DatasetConfig>::err("Missing 'dataset' key in " + path);

        DatasetConfig cfg;
        if (c["kg_snapshot_id"])      cfg.kg_snapshot_id     = c["kg_snapshot_id"].as<std::string>();
        if (c["corpus_chunks_path"])  cfg.corpus_chunks_path = c["corpus_chunks_path"].as<std::string>();
        if (c["max_records"])         cfg.max_records        = c["max_records"].as<int>();
        if (c["max_char_length"]) cfg.max_char_length  = c["max_char_length"].as<int>();
        if (c["split_seed"])      cfg.split_seed       = c["split_seed"].as<int>();
        if (c["max_hop_paths"])   cfg.max_hop_paths    = c["max_hop_paths"].as<int>();
        if (c["refusal_ratio"])   cfg.refusal_ratio    = c["refusal_ratio"].as<double>();
        if (c["generator"])       cfg.generator        = c["generator"].as<std::string>();
        if (YAML::Node sp = c["splits"]) {
            if (sp["train"])      cfg.train_ratio = sp["train"].as<double>();
            if (sp["validation"]) cfg.val_ratio   = sp["validation"].as<double>();
            if (sp["test"])       cfg.test_ratio  = sp["test"].as<double>();
            if (sp["seed"])       cfg.split_seed  = sp["seed"].as<int>();
        }
        return Result<DatasetConfig>::ok(std::move(cfg));
    } catch (const YAML::Exception& e) {
        return Result<DatasetConfig>::err(std::string("YAML error: ") + e.what());
    }
}

// Deterministic split assignment based on record index and seed.
inline std::string assign_split(int idx, int seed, double train, double val) {
    std::mt19937 rng(static_cast<unsigned>(seed + idx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double v = dist(rng);
    if (v < train) return "train";
    if (v < train + val) return "validation";
    return "test";
}

} // namespace slmkg
