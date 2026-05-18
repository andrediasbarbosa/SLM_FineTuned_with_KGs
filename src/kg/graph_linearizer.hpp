#pragma once
#include <string>
#include <vector>
#include "graph_store.hpp"
#include "entity.hpp"
#include "relation.hpp"

namespace slmkg {

// ---------------------------------------------------------------------------
// PathStep — one hop in a multi-hop path
// ---------------------------------------------------------------------------
struct PathStep {
    std::string head_name;
    std::string relation_type;
    std::string tail_name;
    std::string relation_id;
};

// ---------------------------------------------------------------------------
// Triple / hyperedge linearization
// Format: [ENTITY] h1 [ENTITY] h2 [RELATION] r [ENTITY] t1 [ENTITY] t2
// ---------------------------------------------------------------------------
inline std::string linearize_hyperedge(const std::vector<std::string>& heads,
                                        const std::string&              relation,
                                        const std::vector<std::string>& tails)
{
    std::string s;
    for (const auto& h : heads) s += "[ENTITY] " + h + " ";
    s += "[RELATION] " + relation + " ";
    for (const auto& t : tails) s += "[ENTITY] " + t + " ";
    if (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

inline std::string linearize_triple(const std::string& head,
                                     const std::string& relation,
                                     const std::string& tail)
{
    return linearize_hyperedge({head}, relation, {tail});
}

// ---------------------------------------------------------------------------
// Natural-language verbalization
// Binary: "head relation tail."
// Hyperedge: "h1 and h2 relation t1 and t2."
// ---------------------------------------------------------------------------
namespace detail {
inline std::string join_and(const std::vector<std::string>& parts) {
    std::string s;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) s += (i + 1 == parts.size() ? " and " : ", ");
        s += parts[i];
    }
    return s;
}
} // namespace detail

inline std::string verbalize_hyperedge(const std::vector<std::string>& heads,
                                        const std::string&              relation,
                                        const std::vector<std::string>& tails)
{
    return detail::join_and(heads) + " " + relation + " " + detail::join_and(tails) + ".";
}

inline std::string verbalize_triple(const std::string& head,
                                     const std::string& relation,
                                     const std::string& tail)
{
    return verbalize_hyperedge({head}, relation, {tail});
}

// ---------------------------------------------------------------------------
// Multi-hop path linearization
// "[ENTITY] A [RELATION] r1 [ENTITY] B [RELATION] r2 [ENTITY] C"
// ---------------------------------------------------------------------------
inline std::string linearize_path(const std::vector<PathStep>& steps) {
    if (steps.empty()) return {};
    std::string s = "[ENTITY] " + steps[0].head_name;
    for (const auto& step : steps)
        s += " [RELATION] " + step.relation_type + " [ENTITY] " + step.tail_name;
    return s;
}

// "A may be related to C through B, because A r1 B, and B r2 C."
inline std::string verbalize_path(const std::vector<PathStep>& steps) {
    if (steps.empty()) return {};
    if (steps.size() == 1)
        return verbalize_triple(steps[0].head_name, steps[0].relation_type, steps[0].tail_name);

    std::string first = steps.front().head_name;
    std::string last  = steps.back().tail_name;
    std::string via   = steps.front().tail_name;

    std::string reasons;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        if (i > 0) reasons += ", and ";
        reasons += steps[i].head_name + " " + steps[i].relation_type + " " + steps[i].tail_name;
    }
    return first + " may be related to " + last + " through " + via +
           ", because " + reasons + ".";
}

// ---------------------------------------------------------------------------
// Helpers: resolve entity IDs to names from a GraphStore
// ---------------------------------------------------------------------------
inline std::vector<std::string> resolve_names(const std::vector<std::string>& ids,
                                               const GraphStore& store)
{
    std::vector<std::string> names;
    for (const auto& id : ids) {
        bool found = false;
        for (const auto& e : store.entities()) {
            if (e.id == id) { names.push_back(e.canonical_name); found = true; break; }
        }
        if (!found) names.push_back(id); // fallback to id
    }
    return names;
}

inline std::string linearize_relation(const CanonicalRelation& r, const GraphStore& store) {
    return linearize_hyperedge(resolve_names(r.head_ids, store),
                                r.relation_type,
                                resolve_names(r.tail_ids, store));
}

inline std::string verbalize_relation(const CanonicalRelation& r, const GraphStore& store) {
    return verbalize_hyperedge(resolve_names(r.head_ids, store),
                                r.relation_type,
                                resolve_names(r.tail_ids, store));
}

// ---------------------------------------------------------------------------
// Two-hop path finder
// Returns all (r1, r2) pairs where r1.tail ∩ r2.head is non-empty.
// Caps output at max_paths to avoid combinatorial explosion.
// ---------------------------------------------------------------------------
inline std::vector<std::vector<PathStep>> find_two_hop_paths(
        const GraphStore& store,
        int max_paths = 500)
{
    // Build head_entity_id → relations index
    std::unordered_map<std::string, std::vector<const CanonicalRelation*>> head_index;
    for (const auto& r : store.relations())
        for (const auto& h : r.head_ids)
            head_index[h].push_back(&r);

    // Build entity name index
    std::unordered_map<std::string, std::string> id_to_name;
    for (const auto& e : store.entities()) id_to_name[e.id] = e.canonical_name;

    auto name_of = [&](const std::string& id) -> std::string {
        auto it = id_to_name.find(id);
        return it != id_to_name.end() ? it->second : id;
    };

    std::vector<std::vector<PathStep>> paths;
    for (const auto& r1 : store.relations()) {
        if (static_cast<int>(paths.size()) >= max_paths) break;
        for (const auto& t : r1.tail_ids) {
            auto it = head_index.find(t);
            if (it == head_index.end()) continue;
            for (const auto* r2 : it->second) {
                if (static_cast<int>(paths.size()) >= max_paths) break;
                // Avoid trivial self-loops
                if (r1.id == r2->id) continue;
                PathStep s1{name_of(r1.head_ids.empty() ? "" : r1.head_ids[0]),
                             r1.relation_type,
                             name_of(t),
                             r1.id};
                PathStep s2{name_of(t),
                             r2->relation_type,
                             name_of(r2->tail_ids.empty() ? "" : r2->tail_ids[0]),
                             r2->id};
                paths.push_back({s1, s2});
            }
        }
    }
    return paths;
}

} // namespace slmkg
