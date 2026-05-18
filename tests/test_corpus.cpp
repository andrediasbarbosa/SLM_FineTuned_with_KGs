#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <fstream>
#include <filesystem>

#include "corpus/corpus_reader.hpp"
#include "corpus/chunker.hpp"
#include "corpus/corpus_validator.hpp"
#include "corpus/corpus_manifest.hpp"

#ifndef SLMKG_PROJECT_ROOT
#define SLMKG_PROJECT_ROOT "."
#endif

static const std::string ROOT     = SLMKG_PROJECT_ROOT;
static const std::string FIXTURES = ROOT + "/tests/fixtures";

// -------------------------------------------------------------------------
// CorpusReaderTest
// -------------------------------------------------------------------------

TEST_CASE("CorpusReaderTest.LoadsValidJsonl", "[unit]") {
    auto result = slmkg::read_corpus(FIXTURES + "/corpus_valid");
    REQUIRE(result.errors.empty());
    REQUIRE(result.documents.size() == 2);
    CHECK(result.documents[0].id   == "fix_001");
    CHECK(result.documents[1].id   == "fix_002");
    CHECK(!result.documents[0].text.empty());
}

TEST_CASE("CorpusReaderTest.RejectsMalformedJsonl", "[unit]") {
    auto result = slmkg::read_corpus(FIXTURES + "/corpus_malformed");
    CHECK(!result.errors.empty());
}

// -------------------------------------------------------------------------
// CorpusValidatorTest
// -------------------------------------------------------------------------

TEST_CASE("CorpusValidatorTest.RejectsDuplicateDocumentIds", "[unit]") {
    slmkg::DocumentRecord d1;
    d1.id = "dup_id"; d1.source_path = "a"; d1.source_type = "text";
    d1.ingested_at = "2026-01-01T00:00:00Z"; d1.text_hash = "sha256:abc";
    d1.language = "en"; d1.trust_level = "trusted"; d1.text = "hello world";

    slmkg::DocumentRecord d2 = d1;  // same id

    slmkg::QualityConfig q;
    q.fail_on_duplicate_ids = true;
    auto report = slmkg::validate_documents({d1, d2}, q);
    CHECK_FALSE(report.ok());
    REQUIRE(report.errors.size() >= 1);
    CHECK(report.errors[0].find("Duplicate") != std::string::npos);
}

// -------------------------------------------------------------------------
// ChunkerTest
// -------------------------------------------------------------------------

TEST_CASE("ChunkerTest.DoesNotCreateEmptyChunks", "[unit]") {
    slmkg::DocumentRecord doc;
    doc.id = "doc_000001"; doc.text = "Too short.";  // < min_chunk_chars

    slmkg::ChunkingConfig cfg;  // min=300, max=2000, overlap=100
    slmkg::SplitsConfig   splits;
    int counter = 1;
    auto chunks = slmkg::chunk_document(doc, cfg, splits, counter);
    // Text is below min_chunk_chars — expect zero chunks.
    CHECK(chunks.empty());
}

TEST_CASE("ChunkerTest.RespectsMaxChunkSize", "[unit]") {
    slmkg::DocumentRecord doc;
    doc.id = "doc_000001";
    // Build text longer than max_chunk_chars (2000 chars).
    doc.text = std::string(5000, 'x');

    slmkg::ChunkingConfig cfg;  // max=2000
    slmkg::SplitsConfig   splits;
    int counter = 1;
    auto chunks = slmkg::chunk_document(doc, cfg, splits, counter);
    REQUIRE_FALSE(chunks.empty());
    for (const auto& c : chunks)
        CHECK(static_cast<int>(c.text.size()) <= cfg.max_chunk_chars);
}

// -------------------------------------------------------------------------
// CorpusSplitTest
// -------------------------------------------------------------------------

TEST_CASE("CorpusSplitTest.IsDeterministicWithSeed", "[unit]") {
    std::string doc_id = "doc_000001";
    int seed = 42;
    double train = 0.8, val = 0.1;

    std::string s1 = slmkg::detail::assign_split(doc_id, seed, train, val);
    std::string s2 = slmkg::detail::assign_split(doc_id, seed, train, val);
    CHECK(s1 == s2);
    CHECK((s1 == "train" || s1 == "validation" || s1 == "test"));
}

// -------------------------------------------------------------------------
// CorpusManifestTest
// -------------------------------------------------------------------------

TEST_CASE("CorpusManifestTest.HashStableForSameInput", "[unit]") {
    std::string content = "line one\nline two\n";
    std::string h1 = slmkg::sha256_prefixed(content);
    std::string h2 = slmkg::sha256_prefixed(content);
    CHECK(h1 == h2);
    CHECK(h1.substr(0, 7) == "sha256:");
}

// -------------------------------------------------------------------------
// PDF reader tests
// -------------------------------------------------------------------------
TEST_CASE("CorpusReaderTest.LoadsPdf", "[unit]") {
    std::string pdf_path = FIXTURES + "/sample.pdf";
    REQUIRE(std::filesystem::exists(pdf_path));

    // Create a temp directory containing only the PDF.
    auto tmp = std::filesystem::temp_directory_path() / "slmkg_pdf_test";
    std::filesystem::create_directories(tmp);
    std::filesystem::copy_file(pdf_path, tmp / "sample.pdf",
                               std::filesystem::copy_options::overwrite_existing);

    auto result = slmkg::read_corpus(tmp.string());

    REQUIRE(result.errors.empty());
    REQUIRE(result.documents.size() == 1);
    CHECK(result.documents[0].source_type == "pdf");
    CHECK(!result.documents[0].text.empty());
    CHECK(result.documents[0].text_hash.substr(0, 7) == "sha256:");

    std::filesystem::remove_all(tmp);
}

TEST_CASE("CorpusReaderTest.WarnsMissingTextPdf", "[unit]") {
    // A file with a .pdf extension that is actually empty/corrupt should
    // produce a warning and no document (not an error).
    auto tmp = std::filesystem::temp_directory_path() / "slmkg_pdf_corrupt_test";
    std::filesystem::create_directories(tmp);
    std::ofstream f(tmp / "corrupt.pdf");
    f << "this is not a pdf";
    f.close();

    auto result = slmkg::read_corpus(tmp.string());

    CHECK(result.errors.empty());
    CHECK(result.documents.empty());
    CHECK(!result.warnings.empty());

    std::filesystem::remove_all(tmp);
}
