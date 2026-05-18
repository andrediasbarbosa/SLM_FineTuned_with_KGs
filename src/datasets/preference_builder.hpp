#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include "dataset_config.hpp"
#include "kg/graph_store.hpp"
#include "kg/graph_linearizer.hpp"
#include "negative_sampler.hpp"
#include <nlohmann/json.hpp>

namespace slmkg {
namespace fs = std::filesystem;

struct PreferenceRecord {
    std::string              id;
    std::string              prompt;
    std::string              chosen;
    std::string              rejected;
    std::string              reason;
    std::vector<std::string> evidence_relation_ids;
    std::string              kg_snapshot;
    std::string              split;
    std::string              generator;
};

inline void to_json(nlohmann::json& j, const PreferenceRecord& r) {
    j = {
        {"id",                    r.id},
        {"prompt",                r.prompt},
        {"chosen",                r.chosen},
        {"rejected",              r.rejected},
        {"reason",                r.reason},
        {"evidence_relation_ids", r.evidence_relation_ids},
        {"kg_snapshot",           r.kg_snapshot},
        {"split",                 r.split},
        {"generator",             r.generator},
    };
}

struct PreferenceBuildResult {
    int total   = 0;
    int written = 0;
    int skipped = 0;
};

inline PreferenceBuildResult build_preference_dataset(
        const GraphStore&    store,
        const KgSchema&      schema,
        const DatasetConfig& cfg,
        const std::string&   snapshot_id,
        const std::string&   output_path)
{
    PreferenceBuildResult result;
    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out(output_path);

    int counter = 1;
    auto make_id = [&]() {
        std::ostringstream s;
        s << "pref_" << std::setw(6) << std::setfill('0') << counter++;
        return s.str();
    };

    // Entity name map
    std::unordered_map<std::string, std::string> id_to_name;
    for (const auto& e : store.entities()) id_to_name[e.id] = e.canonical_name;

    // Build relation lookup by ID
    std::unordered_map<std::string, const CanonicalRelation*> rel_by_id;
    for (const auto& r : store.relations()) rel_by_id[r.id] = &r;

    auto neg_result = generate_negatives(store, schema, static_cast<unsigned>(cfg.split_seed + 2));

    for (const auto& neg : neg_result.samples) {
        if (result.total >= cfg.max_records) break;

        const CanonicalRelation* pos_rel = nullptr;
        auto it = rel_by_id.find(neg.source_relation_id);
        if (it != rel_by_id.end()) pos_rel = it->second;
        if (!pos_rel) continue;

        std::string chosen_text  = verbalize_relation(*pos_rel, store);
        std::string rejected_text = neg.linearized;

        std::vector<std::string> head_names, tail_names;
        for (const auto& id : pos_rel->head_ids)
            head_names.push_back(id_to_name.count(id) ? id_to_name.at(id) : id);

        std::string prompt = "Describe the relationship involving: " +
                             (head_names.empty() ? "?" : head_names[0]) + ".";

        int total_chars = static_cast<int>(prompt.size() + chosen_text.size() + rejected_text.size());
        if (total_chars > cfg.max_char_length) { ++result.skipped; continue; }

        PreferenceRecord rec;
        rec.id                    = make_id();
        rec.prompt                = prompt;
        rec.chosen                = chosen_text;
        rec.rejected              = rejected_text;
        rec.reason                = "The rejected answer uses a corrupted fact (" + neg.negative_type +
                                    ") not supported by the knowledge graph.";
        rec.evidence_relation_ids = {pos_rel->id};
        rec.kg_snapshot           = snapshot_id;
        rec.split                 = assign_split(counter, cfg.split_seed, cfg.train_ratio, cfg.val_ratio);
        rec.generator             = cfg.generator;

        nlohmann::json j; to_json(j, rec);
        out << j.dump() << "\n";
        ++result.written;
        ++result.total;
    }

    // Two-hop preference pairs (chosen = multi-hop reasoning, rejected = unsupported shortcut)
    auto paths = find_two_hop_paths(store, cfg.max_hop_paths);
    for (const auto& path : paths) {
        if (result.total >= cfg.max_records) break;
        if (path.size() < 2) continue;

        std::string A = path[0].head_name;
        std::string C = path.back().tail_name;

        PreferenceRecord rec;
        rec.id      = make_id();
        rec.prompt  = "Explain the relationship between '" + A + "' and '" + C + "'.";
        rec.chosen  = verbalize_path(path);
        rec.rejected = A + " directly relates to " + C + ".";
        rec.reason  = "The rejected answer collapses a two-hop path into an unsupported direct relation.";
        for (const auto& step : path) rec.evidence_relation_ids.push_back(step.relation_id);
        rec.kg_snapshot = snapshot_id;
        rec.split       = assign_split(counter, cfg.split_seed, cfg.train_ratio, cfg.val_ratio);
        rec.generator   = cfg.generator;

        int total_chars = static_cast<int>(rec.prompt.size() + rec.chosen.size() + rec.rejected.size());
        if (total_chars > cfg.max_char_length) { ++result.skipped; continue; }

        nlohmann::json j; to_json(j, rec);
        out << j.dump() << "\n";
        ++result.written;
        ++result.total;
    }

    return result;
}

} // namespace slmkg
