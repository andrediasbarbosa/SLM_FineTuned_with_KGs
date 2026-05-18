#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "core/errors.hpp"
#include <yaml-cpp/yaml.h>

namespace slmkg {

struct KgAritySpec {
    int min = 1;
    int max = -1; // -1 = unlimited
};

struct KgEntityTypeSpec {
    std::string              name;
    std::string              description;
    std::vector<std::string> required_fields;
};

struct KgRelationTypeSpec {
    std::string  name;
    std::string  description;
    std::string  domain;
    std::string  range;
    KgAritySpec  head_arity;
    KgAritySpec  tail_arity;
};

struct KgSchema {
    std::string                        schema_version;
    std::vector<KgEntityTypeSpec>      entity_types;
    std::vector<KgRelationTypeSpec>    relation_types;

    bool has_entity_type(const std::string& name) const {
        return entity_type_index_.count(name) > 0;
    }
    bool has_relation_type(const std::string& name) const {
        return relation_type_index_.count(name) > 0;
    }
    const KgRelationTypeSpec* get_relation_type(const std::string& name) const {
        auto it = relation_type_index_.find(name);
        return it != relation_type_index_.end() ? it->second : nullptr;
    }
    const KgEntityTypeSpec* get_entity_type(const std::string& name) const {
        auto it = entity_type_index_.find(name);
        return it != entity_type_index_.end() ? it->second : nullptr;
    }

    void rebuild_index() {
        entity_type_index_.clear();
        relation_type_index_.clear();
        for (const auto& e : entity_types)   entity_type_index_[e.name]   = &e;
        for (const auto& r : relation_types) relation_type_index_[r.name] = &r;
    }

private:
    std::unordered_map<std::string, const KgEntityTypeSpec*>   entity_type_index_;
    std::unordered_map<std::string, const KgRelationTypeSpec*> relation_type_index_;
};

namespace detail {
inline KgAritySpec load_arity(const YAML::Node& n) {
    KgAritySpec a;
    if (!n) return a;
    if (n["min"]) a.min = n["min"].as<int>();
    if (n["max"] && !n["max"].IsNull()) a.max = n["max"].as<int>();
    return a;
}
} // namespace detail

inline Result<KgSchema> load_kg_schema(const std::string& path) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        KgSchema schema;
        schema.schema_version = root["schema_version"]
            ? root["schema_version"].as<std::string>() : "";

        if (YAML::Node et = root["entity_types"]) {
            for (const auto& n : et) {
                KgEntityTypeSpec s;
                s.name        = n["name"]        ? n["name"].as<std::string>()        : "";
                s.description = n["description"] ? n["description"].as<std::string>() : "";
                if (n["required_fields"])
                    for (const auto& f : n["required_fields"])
                        s.required_fields.push_back(f.as<std::string>());
                if (!s.name.empty()) schema.entity_types.push_back(std::move(s));
            }
        }

        if (YAML::Node rt = root["relation_types"]) {
            for (const auto& n : rt) {
                KgRelationTypeSpec s;
                s.name        = n["name"]        ? n["name"].as<std::string>()        : "";
                s.description = n["description"] ? n["description"].as<std::string>() : "";
                s.domain      = n["domain"]      ? n["domain"].as<std::string>()      : "";
                s.range       = n["range"]        ? n["range"].as<std::string>()       : "";
                s.head_arity  = detail::load_arity(n["head_arity"]);
                s.tail_arity  = detail::load_arity(n["tail_arity"]);
                if (!s.name.empty()) schema.relation_types.push_back(std::move(s));
            }
        }

        if (schema.entity_types.empty() && schema.relation_types.empty())
            return Result<KgSchema>::err("Schema has no entity_types or relation_types: " + path);

        schema.rebuild_index();
        return Result<KgSchema>::ok(std::move(schema));
    } catch (const YAML::Exception& e) {
        return Result<KgSchema>::err(std::string("YAML error: ") + e.what());
    }
}

} // namespace slmkg
