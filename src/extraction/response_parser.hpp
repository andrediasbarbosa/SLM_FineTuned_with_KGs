#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include "provider_interface.hpp"
#include "core/hashing.hpp"

namespace slmkg {

struct ParsedExtractionResult {
    ExtractionResponse       response;
    std::vector<std::string> validation_errors;
};

// ---------------------------------------------------------------------------
// Field-normalisation helpers
// ---------------------------------------------------------------------------
namespace detail {

inline std::string first_str(const nlohmann::json& j,
                              std::initializer_list<const char*> keys,
                              const std::string& def = "")
{
    for (const char* k : keys) {
        auto it = j.find(k);
        if (it != j.end() && it->is_string()) return it->get<std::string>();
    }
    return def;
}

inline std::vector<std::string> first_str_array(const nlohmann::json& j,
                                                 std::initializer_list<const char*> keys)
{
    for (const char* k : keys) {
        auto it = j.find(k);
        if (it == j.end()) continue;
        if (it->is_string()) return {it->get<std::string>()};
        if (it->is_array()) {
            std::vector<std::string> v;
            for (const auto& el : *it)
                if (el.is_string()) v.push_back(el.get<std::string>());
            return v;
        }
    }
    return {};
}

} // namespace detail

// ---------------------------------------------------------------------------
// parse_extraction_response
// ---------------------------------------------------------------------------
// Supports two response shapes:
//
//  1. Triples format (preferred):
//       {"triples": [{"head":"A","relation":"r","tail":"B","confidence":0.9,"evidence":"..."}]}
//     Entities are synthesised from unique head/tail strings.
//
//  2. Entities+relations format (legacy):
//       {"entities":[...],"relations":[...]}
//     Accepted field aliases — entities: id/local_id, evidence/evidence_text.
//     Relations: subject/head/head_local_ids, predicate/relation_type, object/tail/tail_local_ids.
//     Cross-reference validation is skipped (schema-agnostic).
//
inline ParsedExtractionResult parse_extraction_response(
        const std::string&                     raw,
        const std::unordered_set<std::string>& allowed_entity_types,
        const std::unordered_set<std::string>& allowed_relation_types,
        const ExtractionRequestMetadata&        metadata,
        const std::string&                      provider,
        const std::string&                      model)
{
    ParsedExtractionResult result;
    auto& resp = result.response;
    resp.provider          = provider;
    resp.model             = model;
    resp.raw_response      = raw;
    resp.raw_response_hash = sha256_prefixed(raw);

    if (raw.empty()) {
        resp.status        = ExtractionStatus::empty_response;
        resp.error_message = "Empty response from provider";
        return result;
    }

    auto j = nlohmann::json::parse(raw, nullptr, false);
    if (j.is_discarded()) {
        resp.status        = ExtractionStatus::invalid_json;
        resp.error_message = "Response is not valid JSON";
        return result;
    }

    if (j.contains("status") && j["status"].is_string() &&
        j["status"].get<std::string>() == "provider_error") {
        resp.status        = ExtractionStatus::provider_error;
        resp.error_message = j.value("message", "Provider error");
        return result;
    }

    // -----------------------------------------------------------------------
    // Format 1: triples array
    // -----------------------------------------------------------------------
    if (j.contains("triples") && j["triples"].is_array()) {
        std::unordered_map<std::string, std::string> name_to_lid;
        int eid = 1;

        auto get_entity_lid = [&](const std::string& name) -> std::string {
            auto it = name_to_lid.find(name);
            if (it != name_to_lid.end()) return it->second;
            std::string lid = "e" + std::to_string(eid++);
            name_to_lid[name] = lid;
            RawExtractionEntity e;
            e.local_id = lid;
            e.name     = name;
            e.type     = "unknown";
            resp.entities.push_back(std::move(e));
            return lid;
        };

        for (const auto& t : j["triples"]) {
            if (!t.is_object()) continue;
            std::string head = detail::first_str(t, {"head", "subject", "from"});
            std::string rel  = detail::first_str(t, {"relation", "relation_type",
                                                      "predicate", "type"});
            std::string tail = detail::first_str(t, {"tail", "object", "to"});

            if (head.empty() || rel.empty() || tail.empty()) {
                result.validation_errors.push_back(
                    "Triple missing head, relation, or tail");
                continue;
            }

            if (!allowed_relation_types.empty() && !allowed_relation_types.count(rel)) {
                result.validation_errors.push_back("Unknown relation type: " + rel);
                continue;
            }

            RawExtractionRelation r;
            r.head_local_ids = {get_entity_lid(head)};
            r.relation_type  = rel;
            r.tail_local_ids = {get_entity_lid(tail)};
            r.evidence_text  = detail::first_str(t, {"evidence", "evidence_text"});
            r.confidence     = t.value("confidence", 0.0);
            resp.relations.push_back(std::move(r));
        }

        resp.status = ExtractionStatus::success;
        return result;
    }

    // -----------------------------------------------------------------------
    // Format 2: entities + relations arrays (legacy / fallback)
    // -----------------------------------------------------------------------
    if (!j.contains("entities") || !j["entities"].is_array()) {
        resp.status        = ExtractionStatus::schema_validation_failed;
        resp.error_message = "Response has neither 'triples' nor 'entities' array";
        return result;
    }
    if (!j.contains("relations") || !j["relations"].is_array()) {
        resp.status        = ExtractionStatus::schema_validation_failed;
        resp.error_message = "Missing or non-array 'relations' field";
        return result;
    }

    int auto_id_counter = 1;

    for (const auto& ej : j["entities"]) {
        if (!ej.is_object()) continue;
        std::string name = detail::first_str(ej, {"name"});
        if (name.empty()) {
            result.validation_errors.push_back("Entity missing 'name' field");
            continue;
        }
        RawExtractionEntity e;
        e.name          = name;
        e.local_id      = detail::first_str(ej, {"local_id", "id"},
                                            "e" + std::to_string(auto_id_counter));
        e.type          = detail::first_str(ej, {"type"}, "unknown");
        e.description   = detail::first_str(ej, {"description"});
        e.evidence_text = detail::first_str(ej, {"evidence_text", "evidence", "text"});
        e.confidence    = ej.value("confidence", 0.0);
        ++auto_id_counter;
        if (ej.contains("aliases") && ej["aliases"].is_array())
            for (const auto& a : ej["aliases"])
                if (a.is_string()) e.aliases.push_back(a.get<std::string>());
        if (!allowed_entity_types.empty() && !allowed_entity_types.count(e.type)) {
            result.validation_errors.push_back("Unknown entity type: " + e.type);
            continue;
        }
        resp.entities.push_back(std::move(e));
    }

    for (const auto& rj : j["relations"]) {
        if (!rj.is_object()) continue;
        std::string rel_type = detail::first_str(
            rj, {"relation_type", "predicate", "type", "label", "relation"});
        if (rel_type.empty()) {
            result.validation_errors.push_back("Relation missing relation type");
            continue;
        }
        auto head = detail::first_str_array(
            rj, {"head_local_ids", "head", "subject", "from", "source"});
        auto tail = detail::first_str_array(
            rj, {"tail_local_ids", "tail", "object", "to", "target"});
        if (head.empty()) {
            result.validation_errors.push_back("Relation '" + rel_type + "' missing head");
            continue;
        }
        if (tail.empty()) {
            result.validation_errors.push_back("Relation '" + rel_type + "' missing tail");
            continue;
        }
        if (!allowed_relation_types.empty() && !allowed_relation_types.count(rel_type)) {
            result.validation_errors.push_back("Unknown relation type: " + rel_type);
            continue;
        }
        RawExtractionRelation r;
        r.relation_type  = rel_type;
        r.head_local_ids = std::move(head);
        r.tail_local_ids = std::move(tail);
        r.evidence_text  = detail::first_str(rj, {"evidence_text", "evidence", "text"});
        r.confidence     = rj.value("confidence", 0.0);
        resp.relations.push_back(std::move(r));
    }

    resp.status = ExtractionStatus::success;
    return result;
}

} // namespace slmkg
