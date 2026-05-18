#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "core/hashing.hpp"
#include <nlohmann/json.hpp>

namespace slmkg {
namespace fs = std::filesystem;

struct DatasetManifest {
    std::string id;
    std::string created_at;
    std::string kg_snapshot_id;
    std::string generator;
    int         total_records    = 0;
    int         train_records    = 0;
    int         val_records      = 0;
    int         test_records     = 0;
    int         skipped_records  = 0;
    int         token_estimate   = 0;   // char_count / 4
    std::string content_hash;
    std::string config_hash;
};

inline DatasetManifest compute_manifest(
        const std::string& dataset_path,
        const std::string& dataset_id,
        const std::string& snapshot_id,
        const std::string& generator,
        const std::string& config_str = "")
{
    DatasetManifest m;
    m.id             = dataset_id;
    m.kg_snapshot_id = snapshot_id;
    m.generator      = generator;
    m.config_hash    = config_str.empty() ? "" : sha256_prefixed(config_str);

    // Timestamp
    std::time_t t = std::time(nullptr);
    std::tm* tm_utc = std::gmtime(&t);
    std::ostringstream ts;
    ts << std::put_time(tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    m.created_at = ts.str();

    if (!fs::exists(dataset_path)) return m;

    std::ifstream f(dataset_path);
    std::ostringstream content_buf;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        content_buf << line << "\n";
        ++m.total_records;
        m.token_estimate += static_cast<int>(line.size()) / 4;

        auto j = nlohmann::json::parse(line, nullptr, false);
        if (!j.is_discarded()) {
            std::string split = j.value("split", "train");
            if      (split == "train")      ++m.train_records;
            else if (split == "validation") ++m.val_records;
            else if (split == "test")       ++m.test_records;
        }
    }
    m.content_hash = sha256_prefixed(content_buf.str());
    return m;
}

inline void write_manifest(const DatasetManifest& m, const std::string& path) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path);
    f << nlohmann::json({
        {"id",              m.id},
        {"created_at",      m.created_at},
        {"kg_snapshot_id",  m.kg_snapshot_id},
        {"generator",       m.generator},
        {"total_records",   m.total_records},
        {"train_records",   m.train_records},
        {"val_records",     m.val_records},
        {"test_records",    m.test_records},
        {"skipped_records", m.skipped_records},
        {"token_estimate",  m.token_estimate},
        {"content_hash",    m.content_hash},
        {"config_hash",     m.config_hash},
    }).dump(2) << "\n";
}

} // namespace slmkg
