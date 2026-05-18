#include <catch2/catch_test_macros.hpp>
#include <fstream>

#include "prompts/prompt_template.hpp"
#include "prompts/prompt_renderer.hpp"
#include "prompts/prompt_hash.hpp"
#include "prompts/prompt_manifest.hpp"

#ifndef SLMKG_PROJECT_ROOT
#define SLMKG_PROJECT_ROOT "."
#endif

static const std::string ROOT     = SLMKG_PROJECT_ROOT;
static const std::string FIXTURES = ROOT + "/tests/fixtures";
static const std::string PROMPTS  = ROOT + "/prompts";

// -------------------------------------------------------------------------
// PromptRendererTest
// -------------------------------------------------------------------------

TEST_CASE("PromptRendererTest.SubstitutesAllVariables", "[unit]") {
    slmkg::PromptTemplate tmpl;
    tmpl.content      = "Doc: {{document_id}}\nChunk: {{chunk_id}}";
    tmpl.placeholders = {"document_id", "chunk_id"};

    std::unordered_map<std::string, std::string> vars = {
        {"document_id", "doc_000001"},
        {"chunk_id",    "chunk_000001"},
    };
    auto result = slmkg::render_prompt(tmpl, vars);
    REQUIRE(result.is_ok());
    CHECK(result.value() == "Doc: doc_000001\nChunk: chunk_000001");
}

TEST_CASE("PromptRendererTest.FailsOnMissingVariable", "[unit]") {
    slmkg::PromptTemplate tmpl;
    tmpl.content      = "{{document_id}} {{chunk_id}}";
    tmpl.placeholders = {"document_id", "chunk_id"};

    std::unordered_map<std::string, std::string> vars = {
        {"document_id", "doc_000001"},
        // chunk_id intentionally missing
    };
    auto result = slmkg::render_prompt(tmpl, vars);
    CHECK(result.is_err());
    CHECK(result.error_message().find("chunk_id") != std::string::npos);
}

TEST_CASE("PromptRendererTest.PreservesChunkText", "[unit]") {
    slmkg::PromptTemplate tmpl;
    tmpl.content      = "Text:\n{{chunk_text}}";
    tmpl.placeholders = {"chunk_text"};

    const std::string raw = "Line one.\nLine two.\nLine three with special chars: <>&\"'";
    std::unordered_map<std::string, std::string> vars = {{"chunk_text", raw}};
    auto result = slmkg::render_prompt(tmpl, vars);
    REQUIRE(result.is_ok());
    CHECK(result.value().find(raw) != std::string::npos);
}

TEST_CASE("PromptRendererTest.RendersSchemaCorrectly", "[unit]") {
    slmkg::PromptTemplate tmpl;
    tmpl.content      = "Schema:\n{{kg_schema}}";
    tmpl.placeholders = {"kg_schema"};

    const std::string schema = "schema_version: kg_schema_v0.1\nentity_types: []";
    std::unordered_map<std::string, std::string> vars = {{"kg_schema", schema}};
    auto result = slmkg::render_prompt(tmpl, vars);
    REQUIRE(result.is_ok());
    CHECK(result.value().find("kg_schema_v0.1") != std::string::npos);
}

// -------------------------------------------------------------------------
// PromptHashTest
// -------------------------------------------------------------------------

TEST_CASE("PromptHashTest.HashStableForSamePrompt", "[unit]") {
    const std::string rendered = "You are extracting a KG.\nDocument: doc_000001";
    CHECK(slmkg::hash_prompt(rendered) == slmkg::hash_prompt(rendered));
}

TEST_CASE("PromptHashTest.HashChangesWhenPromptChanges", "[unit]") {
    const std::string a = "Prompt version A";
    const std::string b = "Prompt version B";
    CHECK(slmkg::hash_prompt(a) != slmkg::hash_prompt(b));
}

// -------------------------------------------------------------------------
// PromptManifestTest
// -------------------------------------------------------------------------

TEST_CASE("PromptManifestTest.RecordsPromptPathsAndHashes", "[unit]") {
    const std::string sys_content  = "You are a KG extractor.";
    const std::string user_content = "Extract from {{chunk_text}}.";
    const std::string rendered     = "Extract from Hello world.";

    std::unordered_map<std::string, std::string> vars = {{"chunk_text", "Hello world"}};

    auto record = slmkg::make_prompt_manifest(
        "kg_extract_promptset_v0.1",
        "prompts/kg_system_prompt_v0.1.txt", sys_content,
        "prompts/kg_user_prompt_v0.1.txt",  user_content,
        rendered, vars);

    CHECK(record.prompt_set_id == "kg_extract_promptset_v0.1");
    CHECK(record.system_prompt_path == "prompts/kg_system_prompt_v0.1.txt");
    CHECK(record.user_prompt_path   == "prompts/kg_user_prompt_v0.1.txt");
    CHECK(record.system_prompt_hash.substr(0, 7)   == "sha256:");
    CHECK(record.user_prompt_hash.substr(0, 7)     == "sha256:");
    CHECK(record.rendered_prompt_hash.substr(0, 7) == "sha256:");
    CHECK(record.template_variables.at("chunk_text") == "Hello world");

    // Verify JSON serialization includes all keys
    auto j = slmkg::to_json(record);
    CHECK(j.contains("prompt_set_id"));
    CHECK(j.contains("system_prompt_hash"));
    CHECK(j.contains("rendered_prompt_hash"));
    CHECK(j.contains("template_variables"));
}
