#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "kg/graph_store.hpp"
#include "kg/schema.hpp"
#include "kg/graph_linearizer.hpp"
#include <nlohmann/json.hpp>

namespace slmkg {

struct NegativeSample {
    std::string              negative_type;       // corruption mode
    std::string              source_relation_id;
    std::vector<std::string> head_ids;
    std::string              relation_type;
    std::vector<std::string> tail_ids;
    std::string              linearized;          // linearized corrupted hyperedge
};

struct NegativeSamplerResult {
    std::vector<NegativeSample> samples;
    int skipped = 0; // collision-exhausted
};

// Key for collision detection: (relation_type, sorted head_ids, sorted tail_ids)
struct HyperedgeKey {
    std::string rel;
    std::vector<std::string> heads; // sorted
    std::vector<std::string> tails; // sorted
    bool operator==(const HyperedgeKey& o) const {
        return rel == o.rel && heads == o.heads && tails == o.tails;
    }
};
struct HyperedgeKeyHash {
    std::size_t operator()(const HyperedgeKey& k) const {
        std::size_t h = std::hash<std::string>{}(k.rel);
        for (const auto& s : k.heads) h ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (h<<6) + (h>>2);
        for (const auto& s : k.tails) h ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};

// ---------------------------------------------------------------------------
// generate_negatives
// ---------------------------------------------------------------------------
inline NegativeSamplerResult generate_negatives(
        const GraphStore& store,
        const KgSchema&   schema,
        unsigned          seed        = 42,
        int               max_retries = 10)
{
    NegativeSamplerResult result;
    if (store.relations().empty()) return result;

    std::mt19937 rng(seed);

    // Build entity lookup by type
    std::unordered_map<std::string, std::vector<std::string>> entities_by_type;
    for (const auto& e : store.entities())
        entities_by_type[e.type].push_back(e.id);

    // All entity ids (fallback when type-matched pool is small)
    std::vector<std::string> all_entity_ids;
    for (const auto& e : store.entities()) all_entity_ids.push_back(e.id);

    // All relation types
    std::vector<std::string> all_rel_types;
    for (const auto& rt : schema.relation_types) all_rel_types.push_back(rt.name);

    // Build collision set from existing relations
    std::unordered_set<HyperedgeKey, HyperedgeKeyHash> existing;
    for (const auto& r : store.relations()) {
        HyperedgeKey k;
        k.rel   = r.relation_type;
        k.heads = r.head_ids; std::sort(k.heads.begin(), k.heads.end());
        k.tails = r.tail_ids; std::sort(k.tails.begin(), k.tails.end());
        existing.insert(k);
    }

    // Entity id → name map for linearization
    std::unordered_map<std::string, std::string> id_to_name;
    for (const auto& e : store.entities()) id_to_name[e.id] = e.canonical_name;

    auto name_of = [&](const std::string& id) -> std::string {
        auto it = id_to_name.find(id);
        return it != id_to_name.end() ? it->second : id;
    };

    auto resolve = [&](const std::vector<std::string>& ids) {
        std::vector<std::string> names;
        for (const auto& id : ids) names.push_back(name_of(id));
        return names;
    };

    auto collision = [&](const std::vector<std::string>& heads,
                         const std::string& rel,
                         const std::vector<std::string>& tails) {
        HyperedgeKey k;
        k.rel = rel;
        k.heads = heads; std::sort(k.heads.begin(), k.heads.end());
        k.tails = tails; std::sort(k.tails.begin(), k.tails.end());
        return existing.count(k) > 0;
    };

    auto rand_entity_of_type = [&](const std::string& type,
                                   const std::vector<std::string>& exclude) -> std::string {
        // Use type-matched pool; fall back to all entities
        const auto& pool = entities_by_type.count(type) && entities_by_type.at(type).size() > 1
                         ? entities_by_type.at(type) : all_entity_ids;
        std::vector<std::string> candidates;
        for (const auto& eid : pool)
            if (std::find(exclude.begin(), exclude.end(), eid) == exclude.end())
                candidates.push_back(eid);
        if (candidates.empty()) return {};
        std::uniform_int_distribution<std::size_t> d(0, candidates.size() - 1);
        return candidates[d(rng)];
    };

    auto rand_relation = [&](const std::string& exclude) -> std::string {
        std::vector<std::string> opts;
        for (const auto& rt : all_rel_types) if (rt != exclude) opts.push_back(rt);
        if (opts.empty()) return {};
        std::uniform_int_distribution<std::size_t> d(0, opts.size() - 1);
        return opts[d(rng)];
    };

    for (const auto& rel : store.relations()) {
        // Determine applicable corruption modes
        const KgRelationTypeSpec* rspec = schema.get_relation_type(rel.relation_type);
        int tail_min = rspec ? rspec->tail_arity.min : 1;
        int tail_max = rspec ? rspec->tail_arity.max : -1;
        int head_min = rspec ? rspec->head_arity.min : 1;

        std::vector<std::string> modes = {"corrupt_tail_entity", "corrupt_relation"};
        if (static_cast<int>(rel.tail_ids.size()) > tail_min)
            modes.push_back("shrink_tails");
        if (tail_max < 0 || static_cast<int>(rel.tail_ids.size()) < tail_max)
            modes.push_back("expand_tails");
        if (static_cast<int>(rel.head_ids.size()) > 1 &&
            static_cast<int>(rel.head_ids.size()) > head_min)
            modes.push_back("shrink_heads");

        std::uniform_int_distribution<std::size_t> mode_dist(0, modes.size() - 1);
        std::string chosen_mode = modes[mode_dist(rng)];

        // Determine range type for tail entity corruption
        std::string range_type = rspec ? rspec->range : "unknown";

        bool accepted = false;
        for (int attempt = 0; attempt < max_retries && !accepted; ++attempt) {
            std::vector<std::string> new_heads = rel.head_ids;
            std::string             new_rel    = rel.relation_type;
            std::vector<std::string> new_tails = rel.tail_ids;

            if (chosen_mode == "corrupt_tail_entity" && !rel.tail_ids.empty()) {
                std::string replacement = rand_entity_of_type(range_type, rel.tail_ids);
                if (replacement.empty()) { chosen_mode = "corrupt_relation"; }
                else {
                    std::uniform_int_distribution<std::size_t> idx(0, new_tails.size()-1);
                    new_tails[idx(rng)] = replacement;
                }
            }
            if (chosen_mode == "corrupt_relation") {
                std::string r2 = rand_relation(rel.relation_type);
                if (r2.empty()) continue;
                new_rel = r2;
            }
            if (chosen_mode == "shrink_tails" && new_tails.size() > 1) {
                std::uniform_int_distribution<std::size_t> idx(0, new_tails.size()-1);
                new_tails.erase(new_tails.begin() + static_cast<std::ptrdiff_t>(idx(rng)));
            }
            if (chosen_mode == "expand_tails") {
                std::string extra = rand_entity_of_type(range_type, new_tails);
                if (!extra.empty()) new_tails.push_back(extra);
            }
            if (chosen_mode == "shrink_heads" && new_heads.size() > 1) {
                std::uniform_int_distribution<std::size_t> idx(0, new_heads.size()-1);
                new_heads.erase(new_heads.begin() + static_cast<std::ptrdiff_t>(idx(rng)));
            }

            if (new_heads.empty() || new_tails.empty()) continue;
            if (collision(new_heads, new_rel, new_tails)) continue;

            NegativeSample ns;
            ns.negative_type        = chosen_mode;
            ns.source_relation_id   = rel.id;
            ns.head_ids             = new_heads;
            ns.relation_type        = new_rel;
            ns.tail_ids             = new_tails;
            ns.linearized           = linearize_hyperedge(resolve(new_heads), new_rel, resolve(new_tails));
            result.samples.push_back(std::move(ns));
            accepted = true;
        }
        if (!accepted) ++result.skipped;
    }

    return result;
}

} // namespace slmkg
