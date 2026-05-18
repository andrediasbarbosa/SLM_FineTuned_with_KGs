#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include "document.hpp"
#include "core/config.hpp"

namespace slmkg {

struct ValidationReport {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    bool ok() const { return errors.empty(); }
};

inline ValidationReport validate_documents(const std::vector<DocumentRecord>& docs,
                                            const QualityConfig& cfg) {
    ValidationReport report;
    std::unordered_set<std::string> seen_ids;

    for (const auto& d : docs) {
        // Required field checks
        if (d.id.empty())          report.errors.push_back("Document missing 'id'");
        if (d.source_path.empty()) report.errors.push_back(d.id + ": missing 'source_path'");
        if (d.source_type.empty()) report.errors.push_back(d.id + ": missing 'source_type'");
        if (d.ingested_at.empty()) report.errors.push_back(d.id + ": missing 'ingested_at'");
        if (d.text_hash.empty())   report.errors.push_back(d.id + ": missing 'text_hash'");
        if (d.language.empty())    report.errors.push_back(d.id + ": missing 'language'");
        if (d.trust_level.empty()) report.errors.push_back(d.id + ": missing 'trust_level'");
        if (d.text.empty())        report.errors.push_back(d.id + ": empty text content");

        // Duplicate ID check
        if (!d.id.empty()) {
            if (cfg.fail_on_duplicate_ids && seen_ids.count(d.id)) {
                report.errors.push_back("Duplicate document ID: " + d.id);
            }
            seen_ids.insert(d.id);
        }

        // Warnings
        if (cfg.warn_missing_author  && d.author.empty())
            report.warnings.push_back(d.id + ": missing 'author'");
        if (cfg.warn_missing_license && d.license.empty())
            report.warnings.push_back(d.id + ": missing 'license'");
    }

    return report;
}

inline ValidationReport validate_chunks(const std::vector<ChunkRecord>& chunks) {
    ValidationReport report;
    std::unordered_set<std::string> seen_ids;

    for (const auto& c : chunks) {
        if (c.id.empty())          report.errors.push_back("Chunk missing 'id'");
        if (c.document_id.empty()) report.errors.push_back(c.id + ": missing 'document_id'");
        if (c.text.empty())        report.errors.push_back(c.id + ": empty chunk text");
        if (c.text_hash.empty())   report.errors.push_back(c.id + ": missing 'text_hash'");

        if (!c.id.empty()) {
            if (seen_ids.count(c.id))
                report.errors.push_back("Duplicate chunk ID: " + c.id);
            seen_ids.insert(c.id);
        }
    }

    return report;
}

} // namespace slmkg
