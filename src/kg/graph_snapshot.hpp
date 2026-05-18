#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <ctime>
#include "graph_store.hpp"
#include "schema.hpp"
#include "core/hashing.hpp"
#include "core/errors.hpp"
#include <nlohmann/json.hpp>

namespace slmkg {
namespace fs = std::filesystem;

struct SnapshotManifest {
    std::string id;
    std::string created_at;
    std::string schema_version;
    int         entity_count    = 0;
    int         relation_count  = 0;
    std::string entities_hash;
    std::string relations_hash;
    std::string snapshot_hash;
};

// Produces a deterministic snapshot:
// 1. Sort entities by id, relations by id.
// 2. Serialize each to a single-line JSONL string.
// 3. SHA256 the concatenated serialization.
// 4. Write entities.jsonl, relations.jsonl, manifest.yaml to output_dir.
inline Result<SnapshotManifest> write_snapshot(
        const GraphStore& store,
        const std::string& snapshot_id,
        const std::string& output_dir,
        const KgSchema& schema)
{
    fs::create_directories(output_dir);

    // Timestamp
    std::time_t t = std::time(nullptr);
    std::tm* tm_utc = std::gmtime(&t);
    std::ostringstream ts_ss;
    ts_ss << std::put_time(tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    const std::string ts = ts_ss.str();

    // Sort copies
    auto entities  = store.entities();
    auto relations = store.relations();
    std::sort(entities.begin(),  entities.end(),
              [](const CanonicalEntity& a, const CanonicalEntity& b){ return a.id < b.id; });
    std::sort(relations.begin(), relations.end(),
              [](const CanonicalRelation& a, const CanonicalRelation& b){ return a.id < b.id; });

    // Serialize + hash entities
    std::ostringstream ent_buf;
    std::ofstream f_ent(output_dir + "/entities.jsonl");
    if (!f_ent.is_open())
        return Result<SnapshotManifest>::err("Cannot write " + output_dir + "/entities.jsonl");
    for (const auto& e : entities) {
        nlohmann::json j; to_json(j, e);
        std::string line = j.dump();
        ent_buf << line << "\n";
        f_ent   << line << "\n";
    }
    const std::string entities_hash = sha256_prefixed(ent_buf.str());

    // Serialize + hash relations
    std::ostringstream rel_buf;
    std::ofstream f_rel(output_dir + "/relations.jsonl");
    if (!f_rel.is_open())
        return Result<SnapshotManifest>::err("Cannot write " + output_dir + "/relations.jsonl");
    for (const auto& r : relations) {
        nlohmann::json j; to_json(j, r);
        std::string line = j.dump();
        rel_buf << line << "\n";
        f_rel   << line << "\n";
    }
    const std::string relations_hash = sha256_prefixed(rel_buf.str());

    const std::string snapshot_hash = sha256_prefixed(ent_buf.str() + rel_buf.str());

    // Write manifest.yaml
    std::ofstream f_manifest(output_dir + "/manifest.yaml");
    if (!f_manifest.is_open())
        return Result<SnapshotManifest>::err("Cannot write " + output_dir + "/manifest.yaml");
    f_manifest
        << "kg_snapshot:\n"
        << "  id: "               << snapshot_id                     << "\n"
        << "  created_at: "       << ts                               << "\n"
        << "  schema_version: "   << schema.schema_version            << "\n"
        << "  entity_count: "     << entities.size()                  << "\n"
        << "  relation_count: "   << relations.size()                 << "\n"
        << "  entities_hash: "    << entities_hash                    << "\n"
        << "  relations_hash: "   << relations_hash                   << "\n"
        << "  snapshot_hash: "    << snapshot_hash                    << "\n";

    SnapshotManifest manifest;
    manifest.id             = snapshot_id;
    manifest.created_at     = ts;
    manifest.schema_version = schema.schema_version;
    manifest.entity_count   = static_cast<int>(entities.size());
    manifest.relation_count = static_cast<int>(relations.size());
    manifest.entities_hash  = entities_hash;
    manifest.relations_hash = relations_hash;
    manifest.snapshot_hash  = snapshot_hash;
    return Result<SnapshotManifest>::ok(std::move(manifest));
}

} // namespace slmkg
