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

struct SftRecord {
    std::string              id;
    std::string              task_type;
    std::string              instruction;
    std::string              input;
    std::vector<std::string> graph_context;
    std::string              output;
    std::vector<std::string> evidence_relation_ids;
    std::string              kg_snapshot;
    std::string              split;
    std::string              generator;
};

inline void to_json(nlohmann::json& j, const SftRecord& r) {
    j = {
        {"id",                    r.id},
        {"task_type",             r.task_type},
        {"instruction",           r.instruction},
        {"input",                 r.input},
        {"graph_context",         r.graph_context},
        {"output",                r.output},
        {"evidence_relation_ids", r.evidence_relation_ids},
        {"kg_snapshot",           r.kg_snapshot},
        {"split",                 r.split},
        {"generator",             r.generator},
    };
}

// T4.6: total char length of a serialised SFT record must not exceed max_char_length
inline int sft_char_estimate(const SftRecord& r) {
    int n = static_cast<int>(r.instruction.size() + r.input.size() + r.output.size());
    for (const auto& ctx : r.graph_context) n += static_cast<int>(ctx.size()) + 1;
    return n;
}

struct SftBuildResult {
    int total   = 0;
    int written = 0;
    int skipped = 0;
};

inline SftBuildResult build_sft_dataset(
        const GraphStore&    store,
        const KgSchema&      schema,
        const DatasetConfig& cfg,
        const std::string&   snapshot_id,
        const std::string&   output_path)
{
    SftBuildResult result;
    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out(output_path);

    int counter = 1;
    auto make_id = [&]() {
        std::ostringstream s;
        s << "sft_" << std::setw(6) << std::setfill('0') << counter++;
        return s.str();
    };

    auto write_record = [&](SftRecord& rec) {
        if (sft_char_estimate(rec) > cfg.max_char_length) { ++result.skipped; return; }
        rec.split = assign_split(counter, cfg.split_seed, cfg.train_ratio, cfg.val_ratio);
        nlohmann::json j; to_json(j, rec);
        out << j.dump() << "\n";
        ++result.written;
        ++result.total;
    };

    // Build entity name map
    std::unordered_map<std::string, std::string> id_to_name;
    for (const auto& e : store.entities()) id_to_name[e.id] = e.canonical_name;
    auto name_of = [&](const std::string& id) {
        auto it = id_to_name.find(id);
        return it != id_to_name.end() ? it->second : id;
    };

    // ---- single_hop_qa ----
    for (const auto& rel : store.relations()) {
        if (result.total >= cfg.max_records) break;
        if (rel.head_ids.empty() || rel.tail_ids.empty()) continue;

        std::string head = name_of(rel.head_ids[0]);
        std::string tail = name_of(rel.tail_ids[0]);
        std::string ctx  = linearize_relation(rel, store);
        std::string ev   = rel.evidence.empty() ? "" : rel.evidence[0].text;

        SftRecord rec;
        rec.id                  = make_id();
        rec.task_type           = "single_hop_qa";
        rec.instruction         = "Answer using the provided graph evidence.";
        rec.input               = "What is the '" + rel.relation_type + "' relationship for " + head + "?";
        rec.graph_context       = {ctx};
        rec.output              = head + " " + rel.relation_type + " " + tail + "." +
                                  (ev.empty() ? "" : " Evidence: " + ev);
        rec.evidence_relation_ids = {rel.id};
        rec.kg_snapshot         = snapshot_id;
        rec.generator           = cfg.generator;
        write_record(rec);
    }

    // ---- multi_hop_qa ----
    auto paths = find_two_hop_paths(store, cfg.max_hop_paths);
    for (const auto& path : paths) {
        if (result.total >= cfg.max_records) break;
        if (path.size() < 2) continue;

        std::string A = path[0].head_name;
        std::string C = path.back().tail_name;

        SftRecord rec;
        rec.id            = make_id();
        rec.task_type     = "multi_hop_qa";
        rec.instruction   = "Answer using the provided graph evidence.";
        rec.input         = "How is '" + A + "' connected to '" + C + "'?";
        for (const auto& step : path)
            rec.graph_context.push_back(linearize_triple(step.head_name, step.relation_type, step.tail_name));
        rec.output        = verbalize_path(path);
        for (const auto& step : path) rec.evidence_relation_ids.push_back(step.relation_id);
        rec.kg_snapshot   = snapshot_id;
        rec.generator     = cfg.generator;
        write_record(rec);
    }

    // ---- missing_evidence_refusal ----
    auto neg_result = generate_negatives(store, schema,
                                         static_cast<unsigned>(cfg.split_seed + 1));
    int refusal_target = static_cast<int>(cfg.max_records * cfg.refusal_ratio);
    int refusals_written = 0;
    for (const auto& neg : neg_result.samples) {
        if (refusals_written >= refusal_target || result.total >= cfg.max_records) break;
        std::vector<std::string> head_names, tail_names;
        for (const auto& id : neg.head_ids) head_names.push_back(id_to_name.count(id) ? id_to_name.at(id) : id);
        for (const auto& id : neg.tail_ids) tail_names.push_back(id_to_name.count(id) ? id_to_name.at(id) : id);

        SftRecord rec;
        rec.id            = make_id();
        rec.task_type     = "missing_evidence_refusal";
        rec.instruction   = "Answer only if the graph supports the claim.";
        rec.input         = "Is it true that " + verbalize_hyperedge(head_names, neg.relation_type, tail_names);
        rec.graph_context = {};
        rec.output        = "The available graph evidence does not support that claim.";
        rec.evidence_relation_ids = {neg.source_relation_id};
        rec.kg_snapshot   = snapshot_id;
        rec.generator     = cfg.generator;
        write_record(rec);
        ++refusals_written;
    }

    // ---- contradiction_detection ----
    for (const auto& neg : neg_result.samples) {
        if (result.total >= cfg.max_records) break;
        std::string positive_ctx;
        for (const auto& r : store.relations())
            if (r.id == neg.source_relation_id) {
                positive_ctx = linearize_relation(r, store); break;
            }

        SftRecord rec;
        rec.id            = make_id();
        rec.task_type     = "contradiction_detection";
        rec.instruction   = "Determine if the following is consistent with the knowledge graph.";
        rec.input         = neg.linearized;
        rec.graph_context = positive_ctx.empty() ? std::vector<std::string>{} : std::vector<std::string>{positive_ctx};
        rec.output        = "No. This is not supported by the graph." +
                            (positive_ctx.empty() ? std::string{} : " The graph shows: " + positive_ctx);
        rec.evidence_relation_ids = {neg.source_relation_id};
        rec.kg_snapshot   = snapshot_id;
        rec.generator     = cfg.generator;
        write_record(rec);
    }

    return result;
}

} // namespace slmkg
