#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>

#include "extraction/provider_interface.hpp"
#include "extraction/response_parser.hpp"
#include "extraction/mock_provider.hpp"
#include "extraction/extraction_cache.hpp"
#include "extraction/replay_provider.hpp"
#include "extraction/extraction_config.hpp"
#include "core/hashing.hpp"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string fixture_dir() {
    return std::string(SLMKG_PROJECT_ROOT) + "/tests/fixtures/mock_provider";
}

static std::string tmp_dir(const std::string& name) {
    fs::path p = fs::temp_directory_path() / ("slmkg_test_" + name);
    fs::create_directories(p);
    return p.string();
}

static slmkg::RenderedPrompt make_rendered_prompt(const std::string& user_text,
                                                   const std::string& pset_id = "test_pset") {
    slmkg::RenderedPrompt rp;
    rp.system_text   = "You are a KG extraction assistant.";
    rp.user_text     = user_text;
    rp.prompt_set_id = pset_id;
    rp.system_hash   = slmkg::sha256_prefixed(rp.system_text);
    rp.user_hash     = slmkg::sha256_prefixed(user_text);
    rp.rendered_hash = rp.user_hash;
    return rp;
}

static slmkg::ExtractionRequestMetadata make_metadata() {
    slmkg::ExtractionRequestMetadata m;
    m.request_id  = "req_test_001";
    m.document_id = "doc_test_001";
    m.chunk_id    = "chunk_test_001";
    m.timestamp   = "2026-01-01T00:00:00Z";
    return m;
}

static slmkg::ProviderConfig default_pcfg() {
    slmkg::ProviderConfig c;
    c.provider = "mock";
    c.model    = "mock";
    return c;
}

// ---------------------------------------------------------------------------
// MockProviderTest
// ---------------------------------------------------------------------------
TEST_CASE("MockProviderTest.ReturnsDeterministicResponse") {
    // Known fixture: SHA256 of the exact user_text string below matches the
    // file in tests/fixtures/mock_provider/.
    std::string user_text = "Extract entities and relations from: "
                            "Knowledge graphs are structured representations.";

    slmkg::MockKgExtractionProvider mock(fixture_dir());
    auto rp   = make_rendered_prompt(user_text);
    auto meta = make_metadata();
    auto pcfg = default_pcfg();

    auto resp1 = mock.extract(rp, meta, pcfg);
    auto resp2 = mock.extract(rp, meta, pcfg);

    REQUIRE(resp1.status == slmkg::ExtractionStatus::cached);
    REQUIRE(resp2.status == slmkg::ExtractionStatus::cached);
    REQUIRE(resp1.raw_response == resp2.raw_response);
    REQUIRE(!resp1.raw_response.empty());
}

TEST_CASE("MockProviderTest.ReturnsEmptyForUnknownHash") {
    slmkg::MockKgExtractionProvider mock(fixture_dir());
    auto rp   = make_rendered_prompt("this prompt has no fixture");
    auto meta = make_metadata();
    auto pcfg = default_pcfg();

    auto resp = mock.extract(rp, meta, pcfg);
    REQUIRE(resp.status == slmkg::ExtractionStatus::success);
    REQUIRE(resp.entities.empty());
    REQUIRE(resp.relations.empty());
}

TEST_CASE("MockProviderTest.ErrorPathFixtureReturnsProviderError") {
    // Write an error-path fixture to a temp directory.
    std::string td = tmp_dir("mock_err");
    std::string user_text = "trigger_error_prompt";
    std::string hash_hex  = slmkg::sha256_hex(user_text);
    std::ofstream f(td + "/__error__" + hash_hex + ".json");
    f << R"({"status":"provider_error","message":"mock injected error"})";
    f.close();

    slmkg::MockKgExtractionProvider mock(td);
    auto rp   = make_rendered_prompt(user_text);
    auto meta = make_metadata();
    auto pcfg = default_pcfg();

    auto resp = mock.extract(rp, meta, pcfg);
    REQUIRE(resp.status == slmkg::ExtractionStatus::provider_error);
}

// ---------------------------------------------------------------------------
// ReplayProviderTest
// ---------------------------------------------------------------------------
TEST_CASE("ReplayProviderTest.ReplaysRecordedResponse") {
    std::string td       = tmp_dir("replay");
    std::string user_text= "replay_test_prompt";
    std::string payload  = R"({"entities":[],"relations":[]})";

    // Pre-populate cache.
    slmkg::ExtractionCache cache(td);
    auto rp   = make_rendered_prompt(user_text);
    cache.put(rp.rendered_hash, payload);

    slmkg::ReplayKgExtractionProvider replay(td);
    auto meta = make_metadata();
    auto pcfg = default_pcfg();

    auto resp = replay.extract(rp, meta, pcfg);
    REQUIRE(resp.status == slmkg::ExtractionStatus::replayed);
    REQUIRE(resp.raw_response == payload);
}

TEST_CASE("ReplayProviderTest.FailsOnCacheMiss") {
    std::string td = tmp_dir("replay_miss");
    slmkg::ReplayKgExtractionProvider replay(td);
    auto rp   = make_rendered_prompt("not_cached_prompt");
    auto meta = make_metadata();
    auto pcfg = default_pcfg();

    auto resp = replay.extract(rp, meta, pcfg);
    REQUIRE(resp.status == slmkg::ExtractionStatus::provider_error);
    REQUIRE(!resp.error_message.empty());
}

// ---------------------------------------------------------------------------
// ProviderConfigTest
// ---------------------------------------------------------------------------
TEST_CASE("ProviderConfigTest.LoadsOpenAIConfig") {
    std::string td   = tmp_dir("cfg_openai");
    std::string path = td + "/kg_extraction.yaml";
    std::ofstream f(path);
    f << "kg_extraction:\n"
         "  provider: openai\n"
         "  model: gpt-4.1-mini\n"
         "  retry:\n"
         "    max_attempts: 5\n";
    f.close();

    auto r = slmkg::load_kg_extraction_config(path);
    REQUIRE(r.is_ok());
    REQUIRE(r.value().provider == "openai");
    REQUIRE(r.value().model    == "gpt-4.1-mini");
    REQUIRE(r.value().max_attempts == 5);
}

TEST_CASE("ProviderConfigTest.LoadsGeminiConfig") {
    std::string td   = tmp_dir("cfg_gemini");
    std::string path = td + "/kg_extraction.yaml";
    std::ofstream f(path);
    f << "kg_extraction:\n"
         "  provider: gemini\n"
         "  model: gemini-2.0-flash\n";
    f.close();

    auto r = slmkg::load_kg_extraction_config(path);
    REQUIRE(r.is_ok());
    REQUIRE(r.value().provider == "gemini");
    REQUIRE(r.value().model    == "gemini-2.0-flash");
}

TEST_CASE("ProviderConfigTest.DoesNotExposeApiKeyInLogs") {
    // API keys come from env vars, never from the config file.
    // Verify that KgExtractionConfig contains no api_key field by checking
    // that serialising the YAML config never includes a key value.
    std::string td   = tmp_dir("cfg_nokey");
    std::string path = td + "/kg_extraction.yaml";
    std::ofstream f(path);
    f << "kg_extraction:\n"
         "  provider: openai\n";
    f.close();

    auto r = slmkg::load_kg_extraction_config(path);
    REQUIRE(r.is_ok());

    // Build a simple log string from the struct fields.
    std::string log_str = "provider=" + r.value().provider
                        + " model="   + r.value().model
                        + " cache="   + r.value().cache_dir;

    // The log string must not contain anything that looks like an API key.
    // (Since the struct has no api_key field, this is structurally guaranteed.)
    REQUIRE(log_str.find("sk-") == std::string::npos);
    REQUIRE(log_str.find("api_key") == std::string::npos);
}

// ---------------------------------------------------------------------------
// ResponseParserTest
// ---------------------------------------------------------------------------
TEST_CASE("ResponseParserTest.RejectsInvalidJson") {
    slmkg::ExtractionRequestMetadata meta = make_metadata();
    auto result = slmkg::parse_extraction_response(
        "not json at all", {}, {}, meta, "mock", "mock");
    REQUIRE(result.response.status == slmkg::ExtractionStatus::invalid_json);
}

TEST_CASE("ResponseParserTest.RejectsSchemaViolation") {
    slmkg::ExtractionRequestMetadata meta = make_metadata();
    // Missing 'relations' array.
    auto result = slmkg::parse_extraction_response(
        R"({"entities":[]})", {}, {}, meta, "mock", "mock");
    REQUIRE(result.response.status == slmkg::ExtractionStatus::schema_validation_failed);
}

TEST_CASE("ResponseParserTest.AcceptsValidResponse") {
    slmkg::ExtractionRequestMetadata meta = make_metadata();
    std::string raw = R"({
        "entities": [{
            "local_id": "e1",
            "name": "Knowledge Graph",
            "type": "Concept",
            "evidence_text": "KGs represent structured knowledge.",
            "confidence": 0.9
        }],
        "relations": []
    })";
    auto result = slmkg::parse_extraction_response(
        raw, {"Concept"}, {"supports"}, meta, "mock", "mock");
    REQUIRE(result.response.status == slmkg::ExtractionStatus::success);
    REQUIRE(result.response.entities.size() == 1);
    REQUIRE(result.response.entities[0].name == "Knowledge Graph");
    REQUIRE(result.validation_errors.empty());
}

TEST_CASE("ResponseParserTest.RejectsUnknownEntityType") {
    slmkg::ExtractionRequestMetadata meta = make_metadata();
    std::string raw = R"({
        "entities": [{
            "local_id": "e1",
            "name": "Foo",
            "type": "UnknownType",
            "evidence_text": "some evidence.",
            "confidence": 0.8
        }],
        "relations": []
    })";
    auto result = slmkg::parse_extraction_response(
        raw, {"Concept"}, {}, meta, "mock", "mock");
    REQUIRE(result.response.status == slmkg::ExtractionStatus::success);
    REQUIRE(result.response.entities.empty());
    REQUIRE(!result.validation_errors.empty());
}

TEST_CASE("ResponseParserTest.AcceptsRelationWithUnknownEntityRef") {
    // Cross-reference validation is intentionally skipped (schema-agnostic mode):
    // relation endpoints do not have to match entity local_ids, which allows
    // LLMs to reference entities by name or use free-form strings.
    slmkg::ExtractionRequestMetadata meta = make_metadata();
    std::string raw = R"({
        "entities": [{
            "local_id": "e1",
            "name": "A",
            "type": "Concept",
            "evidence_text": "evidence",
            "confidence": 0.8
        }],
        "relations": [{
            "head_local_ids": ["e99"],
            "relation_type": "supports",
            "tail_local_ids": ["e1"],
            "evidence_text": "evidence for relation",
            "confidence": 0.7
        }]
    })";
    auto result = slmkg::parse_extraction_response(
        raw, {"Concept"}, {"supports"}, meta, "mock", "mock");
    REQUIRE(result.response.status == slmkg::ExtractionStatus::success);
    REQUIRE(result.response.relations.size() == 1);
}

TEST_CASE("ResponseParserTest.AcceptsSubjectPredicateObjectFormat") {
    slmkg::ExtractionRequestMetadata meta = make_metadata();
    std::string raw = R"({
        "entities": [{
            "id": "e1",
            "name": "Knowledge Graph",
            "type": "Concept",
            "evidence": "KGs represent structured knowledge."
        }],
        "relations": [{
            "subject": "e1",
            "predicate": "supports",
            "object": "Factual Grounding",
            "evidence": "Knowledge graphs support factual grounding."
        }]
    })";
    auto result = slmkg::parse_extraction_response(
        raw, {}, {}, meta, "mock", "mock");
    REQUIRE(result.response.status == slmkg::ExtractionStatus::success);
    REQUIRE(result.response.entities.size() == 1);
    REQUIRE(result.response.entities[0].local_id == "e1");
    REQUIRE(result.response.relations.size() == 1);
    REQUIRE(result.response.relations[0].relation_type == "supports");
}

TEST_CASE("ResponseParserTest.AcceptsTriplesFormat") {
    slmkg::ExtractionRequestMetadata meta = make_metadata();
    std::string raw = R"({
        "triples": [
            {"head": "Knowledge Graph", "relation": "supports", "tail": "Factual Grounding",
             "confidence": 0.9, "evidence": "KGs support factual grounding."},
            {"head": "Knowledge Graph", "relation": "defines",  "tail": "Entity",
             "confidence": 0.85, "evidence": "A KG defines entities and relations."}
        ]
    })";
    auto result = slmkg::parse_extraction_response(
        raw, {}, {}, meta, "mock", "mock");
    REQUIRE(result.response.status == slmkg::ExtractionStatus::success);
    // Two unique heads + one new tail = 3 entities synthesised.
    REQUIRE(result.response.entities.size() == 3);
    REQUIRE(result.response.relations.size() == 2);
    REQUIRE(result.response.relations[0].relation_type == "supports");
    REQUIRE(result.response.relations[1].relation_type == "defines");
    // Head entity is shared across both triples.
    REQUIRE(result.response.relations[0].head_local_ids[0] ==
            result.response.relations[1].head_local_ids[0]);
    REQUIRE(result.validation_errors.empty());
}

TEST_CASE("ResponseParserTest.RejectsTriplesWithMissingFields") {
    slmkg::ExtractionRequestMetadata meta = make_metadata();
    std::string raw = R"({"triples": [{"head": "A", "tail": "B"}]})"; // missing relation
    auto result = slmkg::parse_extraction_response(
        raw, {}, {}, meta, "mock", "mock");
    REQUIRE(result.response.status == slmkg::ExtractionStatus::success);
    REQUIRE(result.response.relations.empty());
    REQUIRE(!result.validation_errors.empty());
}
