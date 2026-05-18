#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <memory>
#include "document.hpp"
#include "core/hashing.hpp"
#include <nlohmann/json.hpp>
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>

namespace slmkg {

struct ReadResult {
    std::vector<DocumentRecord> documents;
    std::vector<std::string>    errors;
    std::vector<std::string>    warnings;
};

namespace detail {

inline std::string now_iso8601() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
    gmtime_r(&t, &tm_utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return buf;
}

inline std::string source_type_from_ext(const std::string& ext) {
    if (ext == ".md")    return "markdown";
    if (ext == ".txt")   return "text";
    if (ext == ".jsonl") return "jsonl";
    if (ext == ".pdf")   return "pdf";
    return "unknown";
}

// Extracts plain text from all pages of a PDF. Returns empty string on failure.
inline std::string extract_pdf_text(const std::string& path) {
    std::unique_ptr<poppler::document> doc(
        poppler::document::load_from_file(path));
    if (!doc) return {};

    std::string text;
    for (int i = 0; i < doc->pages(); ++i) {
        std::unique_ptr<poppler::page> pg(doc->create_page(i));
        if (!pg) continue;
        auto utf8 = pg->text().to_utf8();
        if (utf8.empty()) continue;
        if (!text.empty()) text += "\n\n";
        text.append(utf8.begin(), utf8.end());
    }
    return text;
}

inline std::string doc_id(int n) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "doc_%06d", n);
    return buf;
}

inline std::string read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace detail

// Scans source_root recursively for .md, .txt, .jsonl files (sorted for
// determinism). .md/.txt files become one DocumentRecord each.
// .jsonl files are parsed line-by-line; each line must have at least
// "id" and "text" fields.
inline ReadResult read_corpus(const std::string& source_root) {
    namespace fs = std::filesystem;
    ReadResult result;

    if (!fs::exists(source_root)) {
        result.errors.push_back("Source root does not exist: " + source_root);
        return result;
    }

    // Collect paths sorted for determinism.
    std::vector<fs::path> paths;
    for (const auto& entry : fs::recursive_directory_iterator(source_root)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        if (ext == ".md" || ext == ".txt" || ext == ".jsonl" || ext == ".pdf")
            paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());

    const std::string ts = detail::now_iso8601();
    int doc_counter = 1;

    for (const auto& p : paths) {
        std::string ext = p.extension().string();

        if (ext == ".md" || ext == ".txt") {
            std::string text;
            try {
                text = detail::read_file(p);
            } catch (...) {
                result.errors.push_back("Cannot read file: " + p.string());
                continue;
            }
            if (text.empty()) {
                result.warnings.push_back("Empty file skipped: " + p.string());
                continue;
            }
            DocumentRecord doc;
            doc.id          = detail::doc_id(doc_counter++);
            doc.source_path = p.string();
            doc.source_type = detail::source_type_from_ext(ext);
            doc.title       = p.stem().string();
            doc.ingested_at = ts;
            doc.text_hash   = sha256_prefixed(text);
            doc.text        = std::move(text);
            result.documents.push_back(std::move(doc));

        } else if (ext == ".pdf") {
            std::string text = detail::extract_pdf_text(p.string());
            if (text.empty()) {
                result.warnings.push_back("PDF has no extractable text (image-only or corrupt?): " + p.string());
                continue;
            }
            DocumentRecord doc;
            doc.id          = detail::doc_id(doc_counter++);
            doc.source_path = p.string();
            doc.source_type = "pdf";
            doc.title       = p.stem().string();
            doc.ingested_at = ts;
            doc.text_hash   = sha256_prefixed(text);
            doc.text        = std::move(text);
            result.documents.push_back(std::move(doc));

        } else if (ext == ".jsonl") {
            std::ifstream f(p);
            std::string line;
            int line_num = 0;
            while (std::getline(f, line)) {
                ++line_num;
                if (line.empty()) continue;
                try {
                    auto j = nlohmann::json::parse(line);
                    if (!j.contains("text")) {
                        result.errors.push_back(
                            p.string() + ":" + std::to_string(line_num) + ": missing 'text' field");
                        continue;
                    }
                    DocumentRecord doc;
                    doc.id          = j.contains("id") ? j["id"].get<std::string>()
                                                       : detail::doc_id(doc_counter);
                    doc.source_path = p.string();
                    doc.source_type = "jsonl";
                    doc.title       = j.value("title", p.stem().string());
                    doc.author      = j.value("author", std::string{});
                    doc.created_at  = j.value("created_at", std::string{});
                    doc.ingested_at = ts;
                    doc.language    = j.value("language", std::string{"en"});
                    doc.license     = j.value("license", std::string{});
                    doc.trust_level = j.value("trust_level", std::string{"trusted"});
                    doc.domain      = j.value("domain", std::string{});
                    doc.text        = j["text"].get<std::string>();
                    doc.text_hash   = sha256_prefixed(doc.text);
                    ++doc_counter;
                    result.documents.push_back(std::move(doc));
                } catch (const nlohmann::json::exception&) {
                    result.errors.push_back(
                        p.string() + ":" + std::to_string(line_num) + ": malformed JSON");
                }
            }
        }
    }

    return result;
}

} // namespace slmkg
