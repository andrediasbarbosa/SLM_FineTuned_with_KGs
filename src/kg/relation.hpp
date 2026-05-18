#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace slmkg {

struct Evidence {
    std::string source_id;
    std::string chunk_id;
    std::string text;
};

inline void to_json(nlohmann::json& j, const Evidence& e) {
    j = {{"source_id", e.source_id}, {"chunk_id", e.chunk_id}, {"text", e.text}};
}
inline void from_json(const nlohmann::json& j, Evidence& e) {
    e.source_id = j.value("source_id", std::string{});
    e.chunk_id  = j.value("chunk_id",  std::string{});
    e.text      = j.value("text",      std::string{});
}

struct CanonicalRelation {
    std::string              id;
    std::vector<std::string> head_ids;
    std::string              relation_type;
    std::vector<std::string> tail_ids;
    std::vector<Evidence>    evidence;
    double                   confidence           = 0.0;
    std::string              extraction_provider;
    std::string              extraction_model;
    std::string              prompt_set_id;
    std::string              validation_status    = "unvalidated";
    std::string              created_at;
};

inline void to_json(nlohmann::json& j, const CanonicalRelation& r) {
    nlohmann::json ev = nlohmann::json::array();
    for (const auto& e : r.evidence) { nlohmann::json ej; to_json(ej, e); ev.push_back(ej); }
    j = {
        {"id",                   r.id},
        {"head_ids",             r.head_ids},
        {"relation_type",        r.relation_type},
        {"tail_ids",             r.tail_ids},
        {"evidence",             ev},
        {"confidence",           r.confidence},
        {"extraction_provider",  r.extraction_provider},
        {"extraction_model",     r.extraction_model},
        {"prompt_set_id",        r.prompt_set_id},
        {"validation_status",    r.validation_status},
        {"created_at",           r.created_at},
    };
}

inline void from_json(const nlohmann::json& j, CanonicalRelation& r) {
    r.id                  = j.value("id",                   std::string{});
    r.relation_type       = j.value("relation_type",        std::string{});
    r.confidence          = j.value("confidence",           0.0);
    r.extraction_provider = j.value("extraction_provider",  std::string{});
    r.extraction_model    = j.value("extraction_model",     std::string{});
    r.prompt_set_id       = j.value("prompt_set_id",        std::string{});
    r.validation_status   = j.value("validation_status",    std::string{"unvalidated"});
    r.created_at          = j.value("created_at",           std::string{});
    if (j.contains("head_ids") && j["head_ids"].is_array())
        r.head_ids = j["head_ids"].get<std::vector<std::string>>();
    if (j.contains("tail_ids") && j["tail_ids"].is_array())
        r.tail_ids = j["tail_ids"].get<std::vector<std::string>>();
    if (j.contains("evidence") && j["evidence"].is_array())
        for (const auto& ej : j["evidence"]) {
            Evidence e; from_json(ej, e); r.evidence.push_back(std::move(e));
        }
}

} // namespace slmkg
