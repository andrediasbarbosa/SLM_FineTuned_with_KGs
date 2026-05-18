#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include "entity.hpp"
#include "relation.hpp"
#include "graph_store.hpp"
#include "schema.hpp"

namespace slmkg {

struct KgValidationReport {
    int total_entities    = 0;
    int total_relations   = 0;
    int valid_entities    = 0;
    int valid_relations   = 0;
    int rejected_entities = 0;
    int rejected_relations= 0;
    std::vector<std::string> entity_errors;
    std::vector<std::string> relation_errors;

    bool has_hard_errors() const {
        return rejected_entities > 0 || rejected_relations > 0;
    }
};

// Validates all entities and relations in the store against the schema.
// Updates validation_status on each record in-place.
// Returns a KgValidationReport summarising the results.
//
// When strict_types = false (default), unknown types are flagged as warnings
// but do not count as hard errors; the record is still marked auto_validated.
// When strict_types = true, unknown types are hard errors.
inline KgValidationReport validate_graph(GraphStore& store,
                                       const KgSchema& schema,
                                       bool strict_types = false)
{
    KgValidationReport report;

    // Build entity-ID set for endpoint validation.
    std::unordered_set<std::string> entity_ids;
    for (const auto& e : store.entities()) entity_ids.insert(e.id);

    // ---- Validate entities ----
    for (auto& e : store.entities()) {
        ++report.total_entities;
        std::vector<std::string> errs;

        if (e.canonical_name.empty())
            errs.push_back("Entity " + e.id + ": empty canonical_name");

        if (!schema.has_entity_type(e.type)) {
            std::string msg = "Entity " + e.id + " (" + e.canonical_name +
                              "): unknown type '" + e.type + "'";
            if (strict_types) errs.push_back(msg);
            else report.entity_errors.push_back("[WARN] " + msg);
        }

        if (errs.empty()) {
            e.validation_status = "auto_validated";
            ++report.valid_entities;
        } else {
            e.validation_status = "auto_rejected";
            ++report.rejected_entities;
            for (auto& err : errs) report.entity_errors.push_back(err);
        }
    }

    // ---- Validate relations ----
    for (auto& r : store.relations()) {
        ++report.total_relations;
        std::vector<std::string> errs;

        if (r.relation_type.empty())
            errs.push_back("Relation " + r.id + ": empty relation_type");
        if (r.head_ids.empty())
            errs.push_back("Relation " + r.id + ": no head_ids");
        if (r.tail_ids.empty())
            errs.push_back("Relation " + r.id + ": no tail_ids");

        const KgRelationTypeSpec* rspec = schema.get_relation_type(r.relation_type);
        if (!rspec) {
            std::string msg = "Relation " + r.id + ": unknown type '" + r.relation_type + "'";
            if (strict_types) errs.push_back(msg);
            else report.relation_errors.push_back("[WARN] " + msg);
        } else {
            // Head arity check
            int hn = static_cast<int>(r.head_ids.size());
            if (hn < rspec->head_arity.min)
                errs.push_back("Relation " + r.id + ": too few head_ids (min "
                               + std::to_string(rspec->head_arity.min) + ")");
            if (rspec->head_arity.max > 0 && hn > rspec->head_arity.max)
                errs.push_back("Relation " + r.id + ": too many head_ids (max "
                               + std::to_string(rspec->head_arity.max) + ")");

            // Tail arity check
            int tn = static_cast<int>(r.tail_ids.size());
            if (tn < rspec->tail_arity.min)
                errs.push_back("Relation " + r.id + ": too few tail_ids (min "
                               + std::to_string(rspec->tail_arity.min) + ")");
            if (rspec->tail_arity.max > 0 && tn > rspec->tail_arity.max)
                errs.push_back("Relation " + r.id + ": too many tail_ids (max "
                               + std::to_string(rspec->tail_arity.max) + ")");

            // Domain / range checks (warn only — soft constraint)
            if (!rspec->domain.empty()) {
                for (const auto& hid : r.head_ids) {
                    auto it = std::find_if(store.entities().begin(), store.entities().end(),
                                           [&](const CanonicalEntity& e){ return e.id == hid; });
                    if (it != store.entities().end() && it->type != rspec->domain) {
                        report.relation_errors.push_back(
                            "[WARN] Relation " + r.id + " head " + hid +
                            " type '" + it->type + "' != domain '" + rspec->domain + "'");
                    }
                }
            }
            if (!rspec->range.empty()) {
                for (const auto& tid : r.tail_ids) {
                    auto it = std::find_if(store.entities().begin(), store.entities().end(),
                                           [&](const CanonicalEntity& e){ return e.id == tid; });
                    if (it != store.entities().end() && it->type != rspec->range) {
                        report.relation_errors.push_back(
                            "[WARN] Relation " + r.id + " tail " + tid +
                            " type '" + it->type + "' != range '" + rspec->range + "'");
                    }
                }
            }
        }

        if (errs.empty()) {
            r.validation_status = "auto_validated";
            ++report.valid_relations;
        } else {
            r.validation_status = "auto_rejected";
            ++report.rejected_relations;
            for (auto& err : errs) report.relation_errors.push_back(err);
        }
    }

    return report;
}

} // namespace slmkg
