#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace slmkg {

struct CanonicalEntity {
    std::string              id;
    std::string              type;
    std::string              canonical_name;
    std::vector<std::string> aliases;
    std::string              description;
    std::vector<std::string> source_ids;
    std::vector<std::string> chunk_ids;
    double                   confidence        = 0.0;
    std::string              validation_status = "unvalidated";
    std::string              created_at;
    std::string              updated_at;
};

inline void to_json(nlohmann::json& j, const CanonicalEntity& e) {
    j = {
        {"id",                e.id},
        {"type",              e.type},
        {"canonical_name",    e.canonical_name},
        {"aliases",           e.aliases},
        {"description",       e.description},
        {"source_ids",        e.source_ids},
        {"chunk_ids",         e.chunk_ids},
        {"confidence",        e.confidence},
        {"validation_status", e.validation_status},
        {"created_at",        e.created_at},
        {"updated_at",        e.updated_at},
    };
}

inline void from_json(const nlohmann::json& j, CanonicalEntity& e) {
    e.id                = j.value("id",                std::string{});
    e.type              = j.value("type",              std::string{});
    e.canonical_name    = j.value("canonical_name",    std::string{});
    e.description       = j.value("description",       std::string{});
    e.confidence        = j.value("confidence",        0.0);
    e.validation_status = j.value("validation_status", std::string{"unvalidated"});
    e.created_at        = j.value("created_at",        std::string{});
    e.updated_at        = j.value("updated_at",        std::string{});
    if (j.contains("aliases")    && j["aliases"].is_array())
        e.aliases    = j["aliases"].get<std::vector<std::string>>();
    if (j.contains("source_ids") && j["source_ids"].is_array())
        e.source_ids = j["source_ids"].get<std::vector<std::string>>();
    if (j.contains("chunk_ids")  && j["chunk_ids"].is_array())
        e.chunk_ids  = j["chunk_ids"].get<std::vector<std::string>>();
}

} // namespace slmkg
