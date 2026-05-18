#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <unordered_set>
#include "dataset_config.hpp"
#include "kg/graph_store.hpp"
#include "kg/graph_linearizer.hpp"
#include <nlohmann/json.hpp>

namespace slmkg {
namespace fs = std::filesystem;

struct EvalRecord {
    std::string              id;
    std::string              task_type;
    std::string              instruction;
    std::string              input;
    std::string              expected_output;
    std::vector<std::string> evidence_relation_ids;
    std::string              kg_snapshot;
    std::string              split;
    std::string              generator;
};

inline void to_json(nlohmann::json& j, const EvalRecord& r) {
    j = {
        {"id",                    r.id},
        {"task_type",             r.task_type},
        {"instruction",           r.instruction},
        {"input",                 r.input},
        {"expected_output",       r.expected_output},
        {"evidence_relation_ids", r.evidence_relation_ids},
        {"kg_snapshot",           r.kg_snapshot},
        {"split",                 r.split},
        {"generator",             r.generator},
    };
}

struct EvalBuildResult {
    int total_held_out = 0;
    int written        = 0;
    int skipped        = 0;
};

// Holds out the "test" split of relations (hash-based, stable across runs) and
// generates evaluation records from them.
// T4.8: held-out slice is determined before training data generation; same seed
//        as split_seed guarantees no overlap with train/validation records.
inline EvalBuildResult build_eval_dataset(
        const GraphStore&    store,
        const DatasetConfig& cfg,
        const std::string&   snapshot_id,
        const std::string&   output_path)
{
    EvalBuildResult result;
    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out(output_path);

    int counter = 1;
    auto make_id = [&]() {
        std::ostringstream s;
        s << "eval_" << std::setw(6) << std::setfill('0') << counter++;
        return s.str();
    };

    // Entity name map
    std::unordered_map<std::string, std::string> id_to_name;
    for (const auto& e : store.entities()) id_to_name[e.id] = e.canonical_name;

    // Hold out relations whose split is "test" (determined by hash of relation index)
    int rel_idx = 0;
    for (const auto& rel : store.relations()) {
        std::string split = assign_split(rel_idx++, cfg.split_seed,
                                         cfg.train_ratio, cfg.val_ratio);
        if (split != "test") continue;
        ++result.total_held_out;
        if (result.written >= cfg.max_records) { ++result.skipped; continue; }
        if (rel.head_ids.empty() || rel.tail_ids.empty()) continue;

        std::string head = id_to_name.count(rel.head_ids[0]) ? id_to_name.at(rel.head_ids[0]) : rel.head_ids[0];
        std::string tail = id_to_name.count(rel.tail_ids[0]) ? id_to_name.at(rel.tail_ids[0]) : rel.tail_ids[0];
        std::string ctx  = linearize_relation(rel, store);

        // Single-hop factual QA eval record
        EvalRecord rec;
        rec.id                    = make_id();
        rec.task_type             = "single_hop_qa";
        rec.instruction           = "Answer using the provided graph evidence.";
        rec.input                 = "What is the '" + rel.relation_type + "' relationship for " + head + "?";
        rec.expected_output       = head + " " + rel.relation_type + " " + tail + ".";
        rec.evidence_relation_ids = {rel.id};
        rec.kg_snapshot           = snapshot_id;
        rec.split                 = "test";
        rec.generator             = cfg.generator;

        int char_est = static_cast<int>(rec.input.size() + rec.expected_output.size());
        if (char_est > cfg.max_char_length) { ++result.skipped; continue; }

        nlohmann::json j; to_json(j, rec);
        out << j.dump() << "\n";
        ++result.written;
    }

    return result;
}

} // namespace slmkg
