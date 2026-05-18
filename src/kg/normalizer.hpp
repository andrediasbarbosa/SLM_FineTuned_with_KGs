#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include "entity.hpp"
#include "relation.hpp"
#include "graph_store.hpp"
#include <nlohmann/json.hpp>

namespace slmkg {

// ---------------------------------------------------------------------------
// String normalisation: lowercase + collapse whitespace
// ---------------------------------------------------------------------------
inline std::string normalize_name(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool last_space = true; // trim leading
    for (unsigned char c : s) {
        if (std::isspace(c)) {
            if (!last_space) out += ' ';
            last_space = true;
        } else {
            out += static_cast<char>(std::tolower(c));
            last_space = false;
        }
    }
    // trim trailing space
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

struct NormalizationResult {
    GraphStore  store;
    int         candidates_read    = 0;
    int         entities_created   = 0;
    int         entities_merged    = 0;  // duplicate names collapsed
    int         relations_created  = 0;
    int         relations_skipped  = 0;  // endpoints not in canonical set
    std::vector<std::string> warnings;
};

// ---------------------------------------------------------------------------
// normalize_candidates
//
// Reads candidate_entities.jsonl and candidate_relations.jsonl from
// candidate_dir, deduplicates entities by normalized name, maps
// head/tail candidate IDs to canonical entity IDs, and produces a
// GraphStore ready for validation.
// ---------------------------------------------------------------------------
inline NormalizationResult normalize_candidates(const std::string& candidate_dir) {
    NormalizationResult result;

    // -- timestamp --
    std::time_t t = std::time(nullptr);
    std::tm* tm_utc = std::gmtime(&t);
    std::ostringstream ts_ss;
    ts_ss << std::put_time(tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    const std::string ts = ts_ss.str();

    // -- entity ID counter --
    int ent_counter = 1;
    int rel_counter = 1;

    auto make_ent_id = [&]() {
        std::ostringstream s;
        s << "ent_" << std::setw(6) << std::setfill('0') << ent_counter++;
        return s.str();
    };
    auto make_rel_id = [&]() {
        std::ostringstream s;
        s << "rel_" << std::setw(6) << std::setfill('0') << rel_counter++;
        return s.str();
    };

    // Maps for entity normalisation
    // canonical_key (normalize_name) -> index in result.store.entities()
    std::unordered_map<std::string, std::size_t> key_to_idx;
    // candidate_id -> canonical entity id
    std::unordered_map<std::string, std::string> cand_to_ent;

    // ---- Pass 1: read candidate entities ----
    std::string ent_path = candidate_dir + "/candidate_entities.jsonl";
    {
        std::ifstream f(ent_path);
        if (!f.is_open()) {
            result.warnings.push_back("Cannot open " + ent_path);
        } else {
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty()) continue;
                auto j = nlohmann::json::parse(line, nullptr, false);
                if (j.is_discarded()) continue;
                ++result.candidates_read;

                std::string cand_id   = j.value("candidate_id", "");
                std::string name      = j.value("name", "");
                std::string type      = j.value("type", "unknown");
                std::string desc      = j.value("description", "");
                std::string doc_id    = j.value("document_id", "");
                std::string chunk_id  = j.value("chunk_id", "");
                double      conf      = j.value("confidence", 0.0);

                if (name.empty()) continue;

                std::string key = normalize_name(name);
                auto it = key_to_idx.find(key);

                if (it == key_to_idx.end()) {
                    // New canonical entity
                    CanonicalEntity e;
                    e.id                = make_ent_id();
                    e.type              = type;
                    e.canonical_name    = key;
                    e.description       = desc;
                    e.confidence        = conf;
                    e.validation_status = "unvalidated";
                    e.created_at        = ts;
                    e.updated_at        = ts;
                    if (!doc_id.empty())   e.source_ids.push_back(doc_id);
                    if (!chunk_id.empty()) e.chunk_ids.push_back(chunk_id);

                    // Original name as alias if different from canonical key
                    if (name != key) e.aliases.push_back(name);

                    // Aliases from candidate record
                    if (j.contains("aliases") && j["aliases"].is_array())
                        for (const auto& a : j["aliases"])
                            if (a.is_string()) {
                                std::string alias = a.get<std::string>();
                                if (alias != key) e.aliases.push_back(alias);
                            }

                    std::size_t idx = result.store.entities().size();
                    result.store.add_entity(std::move(e));
                    key_to_idx[key] = idx;
                    if (!cand_id.empty()) cand_to_ent[cand_id] = result.store.entities()[idx].id;
                    ++result.entities_created;
                } else {
                    // Merge into existing canonical entity
                    auto& e = result.store.entities()[it->second];
                    ++result.entities_merged;

                    // Merge provenance
                    auto push_unique = [](std::vector<std::string>& v, const std::string& s) {
                        if (!s.empty() && std::find(v.begin(), v.end(), s) == v.end())
                            v.push_back(s);
                    };
                    push_unique(e.source_ids, doc_id);
                    push_unique(e.chunk_ids,  chunk_id);

                    // Merge aliases
                    if (name != key) {
                        std::string alias_norm = normalize_name(name);
                        push_unique(e.aliases, alias_norm != key ? name : "");
                    }
                    if (j.contains("aliases") && j["aliases"].is_array())
                        for (const auto& a : j["aliases"])
                            if (a.is_string()) push_unique(e.aliases, a.get<std::string>());

                    // Average confidence
                    e.confidence = (e.confidence + conf) / 2.0;
                    e.updated_at = ts;

                    if (!cand_id.empty()) cand_to_ent[cand_id] = e.id;
                }
            }
        }
    }

    // ---- Pass 2: read candidate relations ----
    std::string rel_path = candidate_dir + "/candidate_relations.jsonl";
    {
        std::ifstream f(rel_path);
        if (!f.is_open()) {
            result.warnings.push_back("Cannot open " + rel_path);
        } else {
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty()) continue;
                auto j = nlohmann::json::parse(line, nullptr, false);
                if (j.is_discarded()) continue;

                std::string rel_type  = j.value("relation_type", "");
                std::string doc_id    = j.value("document_id", "");
                std::string chunk_id  = j.value("chunk_id", "");
                std::string ev_text   = j.value("evidence_text", "");
                double      conf      = j.value("confidence", 0.0);
                std::string provider  = j.value("provider", "");
                std::string model     = j.value("model", "");
                std::string pset_id   = j.value("prompt_set_id", "");

                if (rel_type.empty()) continue;

                // Map head candidate IDs to canonical entity IDs
                std::vector<std::string> head_ids, tail_ids;
                auto map_ids = [&](const char* field, std::vector<std::string>& out) {
                    if (!j.contains(field) || !j[field].is_array()) return;
                    for (const auto& id : j[field]) {
                        if (!id.is_string()) continue;
                        auto it = cand_to_ent.find(id.get<std::string>());
                        if (it != cand_to_ent.end()) out.push_back(it->second);
                        // Unknown candidate IDs are silently dropped (cross-chunk refs)
                    }
                };
                map_ids("head_candidate_ids", head_ids);
                map_ids("tail_candidate_ids", tail_ids);

                if (head_ids.empty() || tail_ids.empty()) {
                    ++result.relations_skipped;
                    continue;
                }

                CanonicalRelation r;
                r.id                  = make_rel_id();
                r.head_ids            = std::move(head_ids);
                r.relation_type       = rel_type;
                r.tail_ids            = std::move(tail_ids);
                r.confidence          = conf;
                r.extraction_provider = provider;
                r.extraction_model    = model;
                r.prompt_set_id       = pset_id;
                r.validation_status   = "unvalidated";
                r.created_at          = ts;

                if (!ev_text.empty())
                    r.evidence.push_back({doc_id, chunk_id, ev_text});

                result.store.add_relation(std::move(r));
                ++result.relations_created;
            }
        }
    }

    return result;
}

} // namespace slmkg
