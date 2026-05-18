#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include "dataset_config.hpp"
#include "corpus/document.hpp"
#include "kg/graph_store.hpp"
#include "kg/graph_linearizer.hpp"
#include "negative_sampler.hpp"
#include <nlohmann/json.hpp>

namespace slmkg {
namespace fs = std::filesystem;

struct PretrainRecord {
    std::string              id;
    std::string              record_type; // natural_text|graph_linearized|graph_verbalized|...
    std::string              text;
    std::string              split;
    std::string              kg_snapshot;
    std::vector<std::string> source_relation_ids;
    std::string              generator;
};

inline void to_json(nlohmann::json& j, const PretrainRecord& r) {
    j = {
        {"id",                   r.id},
        {"record_type",          r.record_type},
        {"text",                 r.text},
        {"split",                r.split},
        {"kg_snapshot",          r.kg_snapshot},
        {"source_relation_ids",  r.source_relation_ids},
        {"generator",            r.generator},
    };
}

struct PretrainBuildResult {
    int total   = 0;
    int written = 0;
    int skipped = 0; // over length limit
};

inline PretrainBuildResult build_pretrain_dataset(
        const GraphStore&            store,
        const KgSchema&              schema,
        const DatasetConfig&         cfg,
        const std::string&           snapshot_id,
        const std::string&           output_path,
        const std::vector<ChunkRecord>& corpus_chunks = {})
{
    PretrainBuildResult result;
    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out(output_path);

    int counter = 1;
    auto make_id = [&]() {
        std::ostringstream s;
        s << "pretrain_" << std::setw(6) << std::setfill('0') << counter++;
        return s.str();
    };
    // emit_record: writes directly; caller sets rec.split before calling.
    auto emit_record = [&](PretrainRecord& rec) {
        if (static_cast<int>(rec.text.size()) > cfg.max_char_length) {
            ++result.skipped; return;
        }
        nlohmann::json j; to_json(j, rec);
        out << j.dump() << "\n";
        ++result.written;
        ++result.total;
    };
    // write_record: split assigned by counter (used for graph-derived records).
    auto write_record = [&](PretrainRecord& rec) {
        rec.split = assign_split(counter, cfg.split_seed, cfg.train_ratio, cfg.val_ratio);
        emit_record(rec);
    };

    // Raw-text records from corpus chunks (natural-text baseline, no KG).
    for (const auto& chunk : corpus_chunks) {
        if (result.total >= cfg.max_records) break;
        PretrainRecord rec;
        rec.id          = make_id();
        rec.record_type = "raw_text";
        rec.text        = chunk.text;
        rec.split       = chunk.split; // preserve corpus-assigned split
        rec.kg_snapshot = "";          // no KG provenance
        rec.generator   = cfg.generator;
        emit_record(rec);
    }

    // Generate negatives for contradiction samples
    auto neg_result = generate_negatives(store, schema, static_cast<unsigned>(cfg.split_seed));

    for (const auto& rel : store.relations()) {
        if (result.total >= cfg.max_records) break;

        // Graph-linearized record
        {
            PretrainRecord rec;
            rec.id                  = make_id();
            rec.record_type         = "graph_linearized";
            rec.text                = linearize_relation(rel, store);
            rec.kg_snapshot         = snapshot_id;
            rec.source_relation_ids = {rel.id};
            rec.generator           = cfg.generator;
            write_record(rec);
        }
        if (result.total >= cfg.max_records) break;

        // Graph-verbalized record
        {
            PretrainRecord rec;
            rec.id                  = make_id();
            rec.record_type         = "graph_verbalized";
            rec.text                = verbalize_relation(rel, store);
            rec.kg_snapshot         = snapshot_id;
            rec.source_relation_ids = {rel.id};
            rec.generator           = cfg.generator;
            write_record(rec);
        }
    }

    // Two-hop path records
    auto paths = find_two_hop_paths(store, cfg.max_hop_paths);
    for (const auto& path : paths) {
        if (result.total >= cfg.max_records) break;
        PretrainRecord rec;
        rec.id          = make_id();
        rec.record_type = "graph_reasoning_path";
        rec.text        = verbalize_path(path);
        rec.kg_snapshot = snapshot_id;
        for (const auto& step : path) rec.source_relation_ids.push_back(step.relation_id);
        rec.generator   = cfg.generator;
        write_record(rec);
    }

    // Contradiction samples from negatives
    for (const auto& neg : neg_result.samples) {
        if (result.total >= cfg.max_records) break;
        PretrainRecord rec;
        rec.id                  = make_id();
        rec.record_type         = "graph_contradiction";
        rec.text                = "Claim: " + neg.linearized +
                                   "\nLabel: contradicted_by_graph.";
        rec.kg_snapshot         = snapshot_id;
        rec.source_relation_ids = {neg.source_relation_id};
        rec.generator           = cfg.generator;
        write_record(rec);
    }

    return result;
}

} // namespace slmkg
