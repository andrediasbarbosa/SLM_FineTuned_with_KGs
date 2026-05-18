#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include "entity.hpp"
#include "relation.hpp"
#include "core/errors.hpp"
#include <nlohmann/json.hpp>

namespace slmkg {
namespace fs = std::filesystem;

class GraphStore {
public:
    void add_entity(CanonicalEntity e)   { entities_.push_back(std::move(e)); }
    void add_relation(CanonicalRelation r) { relations_.push_back(std::move(r)); }

    const std::vector<CanonicalEntity>&   entities()  const { return entities_; }
    const std::vector<CanonicalRelation>& relations() const { return relations_; }

    std::vector<CanonicalEntity>&   entities()  { return entities_; }
    std::vector<CanonicalRelation>& relations() { return relations_; }

    Result<void> write_entities_jsonl(const std::string& path) const {
        return write_jsonl<CanonicalEntity>(entities_, path);
    }
    Result<void> write_relations_jsonl(const std::string& path) const {
        return write_jsonl<CanonicalRelation>(relations_, path);
    }

    Result<void> load_entities_jsonl(const std::string& path) {
        return load_jsonl<CanonicalEntity>(path, entities_);
    }
    Result<void> load_relations_jsonl(const std::string& path) {
        return load_jsonl<CanonicalRelation>(path, relations_);
    }

    Result<void> write(const std::string& dir) const {
        fs::create_directories(dir);
        auto re = write_entities_jsonl(dir + "/entities.jsonl");
        if (re.is_err()) return re;
        return write_relations_jsonl(dir + "/relations.jsonl");
    }

    Result<void> load(const std::string& dir) {
        auto re = load_entities_jsonl(dir + "/entities.jsonl");
        if (re.is_err()) return re;
        return load_relations_jsonl(dir + "/relations.jsonl");
    }

private:
    std::vector<CanonicalEntity>   entities_;
    std::vector<CanonicalRelation> relations_;

    template<typename T>
    static Result<void> write_jsonl(const std::vector<T>& items,
                                    const std::string& path)
    {
        fs::create_directories(fs::path(path).parent_path());
        std::ofstream f(path);
        if (!f.is_open())
            return Result<void>::err("Cannot write: " + path);
        for (const auto& item : items) {
            nlohmann::json j;
            to_json(j, item);
            f << j.dump() << "\n";
        }
        return Result<void>::ok();
    }

    template<typename T>
    static Result<void> load_jsonl(const std::string& path, std::vector<T>& out) {
        if (!fs::exists(path))
            return Result<void>::err("File not found: " + path);
        std::ifstream f(path);
        std::string line;
        int line_num = 0;
        while (std::getline(f, line)) {
            ++line_num;
            if (line.empty()) continue;
            auto j = nlohmann::json::parse(line, nullptr, false);
            if (j.is_discarded())
                return Result<void>::err(path + ":" + std::to_string(line_num) + ": malformed JSON");
            T item;
            from_json(j, item);
            out.push_back(std::move(item));
        }
        return Result<void>::ok();
    }
};

} // namespace slmkg
