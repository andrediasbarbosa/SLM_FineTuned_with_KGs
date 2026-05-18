#include <iostream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "core/config.hpp"
#include "core/hashing.hpp"
#include "corpus/corpus_reader.hpp"
#include "corpus/chunker.hpp"
#include "corpus/corpus_validator.hpp"
#include "corpus/corpus_manifest.hpp"
#include "prompts/prompt_template.hpp"
#include "prompts/prompt_renderer.hpp"
#include "prompts/prompt_hash.hpp"
#include "prompts/prompt_manifest.hpp"
#include "extraction/extraction_config.hpp"
#include "kg/schema.hpp"
#include "kg/graph_store.hpp"
#include "kg/normalizer.hpp"
#include "kg/graph_validator.hpp"
#include "kg/graph_snapshot.hpp"
#include "datasets/dataset_config.hpp"
#include "datasets/pretrain_builder.hpp"
#include "datasets/sft_builder.hpp"
#include "datasets/preference_builder.hpp"
#include "datasets/eval_builder.hpp"
#include "datasets/dataset_manifest.hpp"
#include "extraction/extraction_pipeline.hpp"
#include "extraction/mock_provider.hpp"
#include "extraction/replay_provider.hpp"
#include "extraction/openai_provider.hpp"
#include "extraction/gemini_provider.hpp"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Minimal flag parser: --key value or --flag (boolean)
// ---------------------------------------------------------------------------
struct Flags {
    std::unordered_map<std::string, std::string> map;

    static Flags parse(int argc, char* argv[], int start = 2) {
        Flags f;
        for (int i = start; i < argc; ++i) {
            std::string a = argv[i];
            if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
                std::string key = a.substr(2);
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    f.map[key] = argv[++i];
                } else {
                    f.map[key] = "1";
                }
            }
        }
        return f;
    }

    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = map.find(k);
        return it != map.end() ? it->second : def;
    }
    bool has(const std::string& k) const { return map.count(k) > 0; }
};

static void print_errors(const std::vector<std::string>& errs,
                          const std::vector<std::string>& warns) {
    for (const auto& w : warns) std::cerr << "[WARN] " << w << "\n";
    for (const auto& e : errs)  std::cerr << "[ERROR] " << e << "\n";
}

// ---------------------------------------------------------------------------
// corpus ingest
// ---------------------------------------------------------------------------
static int cmd_corpus_ingest(const Flags& f) {
    std::string config_path = f.get("config", "configs/corpus.yaml");
    auto cfg_result = slmkg::load_corpus_config(config_path);
    if (cfg_result.is_err()) {
        std::cerr << cfg_result.error_message() << "\n";
        return 1;
    }
    slmkg::CorpusConfig cfg = cfg_result.value();

    if (f.has("input"))  cfg.source_root = f.get("input");
    if (f.has("output")) cfg.output_root = f.get("output");

    if (cfg.source_root.empty()) { std::cerr << "--input or corpus.source_root required\n"; return 1; }
    if (cfg.output_root.empty()) { std::cerr << "--output or corpus.output_root required\n"; return 1; }

    auto read = slmkg::read_corpus(cfg.source_root);
    print_errors(read.errors, read.warnings);
    if (!read.errors.empty() && cfg.quality.fail_on_malformed_jsonl) {
        std::cerr << "Ingestion aborted due to read errors.\n";
        return 1;
    }

    auto doc_report = slmkg::validate_documents(read.documents, cfg.quality);
    print_errors(doc_report.errors, doc_report.warnings);
    if (!doc_report.ok()) { std::cerr << "Document validation failed.\n"; return 1; }

    // Chunk all documents
    std::vector<slmkg::ChunkRecord> all_chunks;
    int next_chunk_id = 1;
    for (auto& doc : read.documents) {
        doc.split = slmkg::detail::assign_split(doc.id,
                                                 cfg.splits.seed,
                                                 cfg.splits.train,
                                                 cfg.splits.validation);
        auto chunks = slmkg::chunk_document(doc, cfg.chunking, cfg.splits, next_chunk_id);
        all_chunks.insert(all_chunks.end(), chunks.begin(), chunks.end());
    }

    auto chunk_report = slmkg::validate_chunks(all_chunks);
    print_errors(chunk_report.errors, chunk_report.warnings);
    if (!chunk_report.ok()) { std::cerr << "Chunk validation failed.\n"; return 1; }

    auto write_result = slmkg::write_corpus_jsonl(cfg.output_root, read.documents, all_chunks);
    if (write_result.is_err()) { std::cerr << write_result.error_message() << "\n"; return 1; }

    const std::string chunks_hash = write_result.value();
    auto stats = slmkg::compute_stats(read.documents, all_chunks);

    if (cfg.manifest.write) {
        auto ts = slmkg::detail::now_iso8601();
        auto mr = slmkg::write_manifest(cfg.output_root, cfg, stats, chunks_hash, ts);
        if (mr.is_err()) { std::cerr << mr.error_message() << "\n"; return 1; }
    }

    auto sr = slmkg::write_stats_json(cfg.output_root, stats);
    if (sr.is_err()) { std::cerr << sr.error_message() << "\n"; return 1; }

    std::cout << "Ingested " << stats.document_count << " documents, "
              << stats.chunk_count << " chunks -> " << cfg.output_root << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// corpus validate
// ---------------------------------------------------------------------------
static int cmd_corpus_validate(const Flags& f) {
    std::string corpus_dir = f.get("corpus");
    if (corpus_dir.empty()) { std::cerr << "--corpus required\n"; return 1; }

    // Validate that expected output files are present
    for (const auto& name : {"documents.jsonl", "chunks.jsonl", "corpus_manifest.yaml"}) {
        if (!fs::exists(corpus_dir + "/" + name)) {
            std::cerr << "Missing: " << corpus_dir << "/" << name << "\n";
            return 1;
        }
    }
    std::cout << "Corpus at " << corpus_dir << " looks valid.\n";
    return 0;
}

// ---------------------------------------------------------------------------
// corpus stats
// ---------------------------------------------------------------------------
static int cmd_corpus_stats(const Flags& f) {
    std::string corpus_dir = f.get("corpus");
    std::string out_path   = f.get("output", corpus_dir + "/corpus_stats.json");
    if (corpus_dir.empty()) { std::cerr << "--corpus required\n"; return 1; }

    std::string stats_src = corpus_dir + "/corpus_stats.json";
    if (!fs::exists(stats_src)) { std::cerr << "Stats file not found: " << stats_src << "\n"; return 1; }

    if (out_path != stats_src) {
        fs::copy_file(stats_src, out_path, fs::copy_options::overwrite_existing);
        std::cout << "Stats written to " << out_path << "\n";
    } else {
        std::ifstream f2(stats_src);
        std::cout << f2.rdbuf() << "\n";
    }
    return 0;
}

// ---------------------------------------------------------------------------
// prompt render
// ---------------------------------------------------------------------------
static int cmd_prompt_render(const Flags& f) {
    std::string sys_path    = f.get("system-prompt");
    std::string user_path   = f.get("user-prompt");
    std::string chunk_id    = f.get("chunk-id");
    std::string corpus_dir  = f.get("corpus");
    std::string schema_path = f.get("schema");
    std::string out_path    = f.get("output");

    if (sys_path.empty() || user_path.empty() || chunk_id.empty() || corpus_dir.empty()) {
        std::cerr << "--system-prompt, --user-prompt, --chunk-id, --corpus required\n";
        return 1;
    }

    // Load templates
    auto sys_tmpl_r  = slmkg::load_prompt_template(sys_path);
    auto user_tmpl_r = slmkg::load_prompt_template(user_path);
    if (sys_tmpl_r.is_err())  { std::cerr << sys_tmpl_r.error_message()  << "\n"; return 1; }
    if (user_tmpl_r.is_err()) { std::cerr << user_tmpl_r.error_message() << "\n"; return 1; }

    // Find chunk in chunks.jsonl
    std::string chunk_text;
    std::string document_id;
    {
        std::ifstream cf(corpus_dir + "/chunks.jsonl");
        if (!cf.is_open()) { std::cerr << "Cannot open " << corpus_dir << "/chunks.jsonl\n"; return 1; }
        std::string line;
        while (std::getline(cf, line)) {
            if (line.empty()) continue;
            auto j = nlohmann::json::parse(line, nullptr, false);
            if (!j.is_discarded() && j.value("id", "") == chunk_id) {
                chunk_text  = j.value("text", "");
                document_id = j.value("document_id", "");
                break;
            }
        }
        if (chunk_text.empty()) {
            std::cerr << "Chunk not found: " << chunk_id << "\n";
            return 1;
        }
    }

    // Load schema as raw text
    std::string schema_text;
    if (!schema_path.empty() && fs::exists(schema_path)) {
        std::ifstream sf(schema_path);
        std::ostringstream ss;
        ss << sf.rdbuf();
        schema_text = ss.str();
        if (!schema_text.empty() && schema_text.back() == '\n')
            schema_text.pop_back();
    }

    // Derive allowed types from schema YAML
    std::string allowed_entities, allowed_relations;
    if (!schema_path.empty() && fs::exists(schema_path)) {
        try {
            YAML::Node s = YAML::LoadFile(schema_path);
            std::string ent_list, rel_list;
            if (s["entity_types"]) {
                for (const auto& e : s["entity_types"]) {
                    if (!ent_list.empty()) ent_list += ", ";
                    ent_list += e["name"].as<std::string>();
                }
            }
            if (s["relation_types"]) {
                for (const auto& r : s["relation_types"]) {
                    if (!rel_list.empty()) rel_list += ", ";
                    rel_list += r["name"].as<std::string>();
                }
            }
            allowed_entities  = ent_list;
            allowed_relations = rel_list;
        } catch (...) {}
    }

    std::unordered_map<std::string, std::string> vars = {
        {"document_id",            document_id},
        {"chunk_id",               chunk_id},
        {"chunk_text",             chunk_text},
        {"kg_schema",              schema_text},
        {"allowed_entity_types",   allowed_entities},
        {"allowed_relation_types", allowed_relations},
        {"known_entities",         ""},
        {"extraction_mode",        "entities_and_relations"},
        {"output_schema",          ""},
    };

    auto rendered_r = slmkg::render_prompt(user_tmpl_r.value(), vars);
    if (rendered_r.is_err()) { std::cerr << rendered_r.error_message() << "\n"; return 1; }

    const std::string& rendered = rendered_r.value();
    if (!out_path.empty()) {
        std::ofstream o(out_path);
        o << rendered;
        std::cout << "Rendered prompt written to " << out_path << "\n";
    } else {
        std::cout << rendered;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// prompt hash
// ---------------------------------------------------------------------------
static int cmd_prompt_hash(const Flags& f) {
    std::string sys_path  = f.get("system-prompt");
    std::string user_path = f.get("user-prompt");
    if (sys_path.empty() || user_path.empty()) {
        std::cerr << "--system-prompt and --user-prompt required\n";
        return 1;
    }
    auto read_file = [](const std::string& p) {
        std::ifstream f(p);
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    };
    std::cout << "system: " << slmkg::sha256_prefixed(read_file(sys_path))  << "\n";
    std::cout << "user:   " << slmkg::sha256_prefixed(read_file(user_path)) << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// prompt validate
// ---------------------------------------------------------------------------
static int cmd_prompt_validate(const Flags& f) {
    std::string sys_path  = f.get("system-prompt");
    std::string user_path = f.get("user-prompt");
    if (sys_path.empty() || user_path.empty()) {
        std::cerr << "--system-prompt and --user-prompt required\n";
        return 1;
    }
    auto sys_r  = slmkg::load_prompt_template(sys_path);
    auto user_r = slmkg::load_prompt_template(user_path);
    if (sys_r.is_err())  { std::cerr << sys_r.error_message()  << "\n"; return 1; }
    if (user_r.is_err()) { std::cerr << user_r.error_message() << "\n"; return 1; }

    std::cout << "system prompt:  " << sys_path  << " OK ("
              << sys_r.value().placeholders.size()  << " placeholders)\n";
    std::cout << "user prompt:    " << user_path << " OK ("
              << user_r.value().placeholders.size() << " placeholders)\n";
    return 0;
}

// ---------------------------------------------------------------------------
// kg normalize
// ---------------------------------------------------------------------------
static int cmd_kg_normalize(const Flags& f) {
    std::string input_dir  = f.get("input");
    std::string output_dir = f.get("output");
    if (input_dir.empty() || output_dir.empty()) {
        std::cerr << "--input and --output required\n"; return 1;
    }

    auto result = slmkg::normalize_candidates(input_dir);
    for (const auto& w : result.warnings) std::cerr << "[WARN] " << w << "\n";

    auto wr = result.store.write(output_dir);
    if (wr.is_err()) { std::cerr << wr.error_message() << "\n"; return 1; }

    std::cout << "Normalized: " << result.entities_created << " entities ("
              << result.entities_merged  << " merged), "
              << result.relations_created << " relations ("
              << result.relations_skipped << " skipped) -> " << output_dir << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// kg validate
// ---------------------------------------------------------------------------
static int cmd_kg_validate(const Flags& f) {
    std::string kg_dir      = f.get("kg");
    std::string schema_path = f.get("schema", "configs/kg_schema.yaml");
    std::string out_path    = f.get("output");
    bool        strict      = f.has("strict");

    if (kg_dir.empty()) { std::cerr << "--kg required\n"; return 1; }

    auto schema_r = slmkg::load_kg_schema(schema_path);
    if (schema_r.is_err()) { std::cerr << schema_r.error_message() << "\n"; return 1; }

    slmkg::GraphStore store;
    auto lr = store.load(kg_dir);
    if (lr.is_err()) { std::cerr << lr.error_message() << "\n"; return 1; }

    slmkg::KgValidationReport report = slmkg::validate_graph(store, schema_r.value(), strict);

    // Print summary
    std::cout << "Entities:  " << report.valid_entities   << "/" << report.total_entities
              << " valid, " << report.rejected_entities << " rejected\n";
    std::cout << "Relations: " << report.valid_relations  << "/" << report.total_relations
              << " valid, " << report.rejected_relations << " rejected\n";
    for (const auto& e : report.entity_errors)   std::cerr << e << "\n";
    for (const auto& e : report.relation_errors) std::cerr << e << "\n";

    // Write updated store (validation_status set in-place)
    auto wr = store.write(kg_dir);
    if (wr.is_err()) std::cerr << "[WARN] " << wr.error_message() << "\n";

    // Write JSON report if requested
    if (!out_path.empty()) {
        fs::create_directories(fs::path(out_path).parent_path());
        std::ofstream rpt(out_path);
        rpt << nlohmann::json({
            {"total_entities",    report.total_entities},
            {"valid_entities",    report.valid_entities},
            {"rejected_entities", report.rejected_entities},
            {"total_relations",   report.total_relations},
            {"valid_relations",   report.valid_relations},
            {"rejected_relations",report.rejected_relations},
            {"entity_errors",     report.entity_errors},
            {"relation_errors",   report.relation_errors},
            {"has_hard_errors",   report.has_hard_errors()},
        }).dump(2) << "\n";
        std::cout << "Report written to " << out_path << "\n";
    }

    return report.has_hard_errors() && strict ? 1 : 0;
}

// ---------------------------------------------------------------------------
// kg snapshot
// ---------------------------------------------------------------------------
static int cmd_kg_snapshot(const Flags& f) {
    std::string kg_dir      = f.get("kg");
    std::string snapshot_id = f.get("snapshot-id", "kg_snapshot");
    std::string output_dir  = f.get("output");
    std::string schema_path = f.get("schema", "configs/kg_schema.yaml");

    if (kg_dir.empty() || output_dir.empty()) {
        std::cerr << "--kg and --output required\n"; return 1;
    }

    auto schema_r = slmkg::load_kg_schema(schema_path);
    if (schema_r.is_err()) { std::cerr << schema_r.error_message() << "\n"; return 1; }

    slmkg::GraphStore store;
    auto lr = store.load(kg_dir);
    if (lr.is_err()) { std::cerr << lr.error_message() << "\n"; return 1; }

    auto snap_r = slmkg::write_snapshot(store, snapshot_id, output_dir, schema_r.value());
    if (snap_r.is_err()) { std::cerr << snap_r.error_message() << "\n"; return 1; }

    const auto& m = snap_r.value();
    std::cout << "Snapshot '" << m.id << "': "
              << m.entity_count << " entities, " << m.relation_count << " relations\n"
              << "  hash: " << m.snapshot_hash << "\n"
              << "  -> "   << output_dir << "/manifest.yaml\n";
    return 0;
}

// ---------------------------------------------------------------------------
// kg extract
// ---------------------------------------------------------------------------
static int cmd_kg_extract(const Flags& f) {
    std::string corpus_dir   = f.get("corpus");
    std::string schema_path  = f.get("schema");
    std::string sys_path     = f.get("system-prompt");
    std::string user_path    = f.get("user-prompt");
    std::string output_dir   = f.get("output");
    std::string config_path  = f.get("config", "configs/kg_extraction.yaml");
    std::string record_to    = f.get("record-to");
    bool        use_mock     = f.has("mock");
    bool        use_replay   = f.has("replay");

    if (corpus_dir.empty() || output_dir.empty()) {
        std::cerr << "--corpus and --output required\n";
        return 1;
    }

    auto cfg_r = slmkg::load_kg_extraction_config(config_path);
    if (cfg_r.is_err()) { std::cerr << cfg_r.error_message() << "\n"; return 1; }
    slmkg::KgExtractionConfig cfg = cfg_r.value();

    // CLI flags override config
    if (use_mock)   cfg.provider = "mock";
    if (use_replay) cfg.provider = "replay";
    if (cfg.test_mode) cfg.provider = "mock";

    // Prompt paths: CLI > config > defaults
    if (!sys_path.empty())  cfg.prompts.system_prompt_path = sys_path;
    if (!user_path.empty()) cfg.prompts.user_prompt_path   = user_path;
    if (!schema_path.empty()) cfg.output.schema_path = schema_path;

    // Load prompt templates
    auto sys_r  = slmkg::load_prompt_template(cfg.prompts.system_prompt_path);
    auto user_r = slmkg::load_prompt_template(cfg.prompts.user_prompt_path);
    if (sys_r.is_err())  { std::cerr << sys_r.error_message()  << "\n"; return 1; }
    if (user_r.is_err()) { std::cerr << user_r.error_message() << "\n"; return 1; }

    // Load chunks
    std::vector<slmkg::ChunkRecord> chunks;
    {
        std::ifstream cf(corpus_dir + "/chunks.jsonl");
        if (!cf.is_open()) { std::cerr << "Cannot open " << corpus_dir << "/chunks.jsonl\n"; return 1; }
        std::string line;
        while (std::getline(cf, line)) {
            if (line.empty()) continue;
            auto j = nlohmann::json::parse(line, nullptr, false);
            if (j.is_discarded()) continue;
            slmkg::ChunkRecord c;
            c.id          = j.value("id", "");
            c.document_id = j.value("document_id", "");
            c.text        = j.value("text", "");
            if (!c.id.empty()) chunks.push_back(std::move(c));
        }
    }
    if (chunks.empty()) { std::cerr << "No chunks found in " << corpus_dir << "/chunks.jsonl\n"; return 1; }

    // Build allowed-type sets from schema only when strict_validation is on.
    // When false (default), extraction is schema-agnostic: any type the LLM
    // returns is accepted. Type enforcement happens in Phase 3 normalization.
    std::unordered_set<std::string> allowed_ent_types, allowed_rel_types;
    if (cfg.output.strict_validation &&
        !cfg.output.schema_path.empty() && fs::exists(cfg.output.schema_path)) {
        try {
            YAML::Node s = YAML::LoadFile(cfg.output.schema_path);
            if (s["entity_types"])
                for (const auto& e : s["entity_types"])
                    allowed_ent_types.insert(e["name"].as<std::string>());
            if (s["relation_types"])
                for (const auto& r : s["relation_types"])
                    allowed_rel_types.insert(r["name"].as<std::string>());
        } catch (...) {}
    }

    slmkg::ProviderConfig prov_cfg;
    prov_cfg.provider           = cfg.provider;
    prov_cfg.model              = cfg.model;
    prov_cfg.temperature        = cfg.temperature;
    prov_cfg.max_output_tokens  = cfg.max_output_tokens;
    prov_cfg.max_attempts       = cfg.max_attempts;
    prov_cfg.initial_backoff_ms = cfg.initial_backoff_ms;
    prov_cfg.max_backoff_ms     = cfg.max_backoff_ms;
    prov_cfg.connect_timeout_ms = cfg.connect_timeout_ms;
    prov_cfg.request_timeout_ms = cfg.request_timeout_ms;

    slmkg::ExtractionPipelineConfig pcfg;
    pcfg.cfg                   = cfg;
    pcfg.provider_cfg          = prov_cfg;
    pcfg.allowed_entity_types  = allowed_ent_types;
    pcfg.allowed_relation_types= allowed_rel_types;
    pcfg.output_dir            = output_dir;
    pcfg.record_to_dir         = record_to;

    slmkg::ExtractionCache cache(cfg.cache_dir);

    // Build provider
    std::unique_ptr<slmkg::IKgExtractionProvider> provider;
    if (cfg.provider == "mock") {
        std::string fixture_dir = f.get("fixture-dir", "tests/fixtures/mock_provider");
        provider = std::make_unique<slmkg::MockKgExtractionProvider>(fixture_dir);
    } else if (cfg.provider == "replay") {
        provider = std::make_unique<slmkg::ReplayKgExtractionProvider>(cfg.cache_dir);
    } else if (cfg.provider == "gemini") {
        provider = std::make_unique<slmkg::GeminiKgExtractionProvider>();
    } else {
        provider = std::make_unique<slmkg::OpenAIKgExtractionProvider>();
    }

    auto result = slmkg::run_extraction_pipeline(
        chunks, sys_r.value(), user_r.value(), *provider, cache, pcfg);

    for (const auto& e : result.errors) std::cerr << "[ERROR] " << e << "\n";

    std::cout << "Processed " << result.chunks_processed << " chunks ("
              << result.chunks_skipped    << " skipped from cache), "
              << result.entities_written  << " entities, "
              << result.relations_written << " relations -> "
              << output_dir << "\n";
    if (result.rejected > 0)
        std::cout << "[WARN] " << result.rejected << " candidates rejected (see rejected_candidates.jsonl)\n";

    return result.errors.empty() ? 0 : 1;
}

// ---------------------------------------------------------------------------
// dataset build-*
// ---------------------------------------------------------------------------
static int cmd_dataset_build(const Flags& f, const std::string& kind) {
    std::string kg_dir      = f.get("kg");
    std::string output_path = f.get("output");
    std::string schema_path = f.get("schema", "configs/kg_schema.yaml");
    std::string snapshot_id = f.get("snapshot-id", "kg_domain_v0.1");

    std::string cfg_prefix  = (kind == "eval" ? std::string("eval") :
                               kind == "pretrain" ? std::string("pretrain") : std::string("sft"));
    std::string default_cfg = "configs/" + cfg_prefix + "_dataset.yaml";
    std::string config_path = f.get("config", default_cfg);

    if (kg_dir.empty() || output_path.empty()) {
        std::cerr << "--kg and --output required\n"; return 1;
    }

    auto cfg_r = slmkg::load_dataset_config(config_path);
    if (cfg_r.is_err()) { std::cerr << cfg_r.error_message() << "\n"; return 1; }
    slmkg::DatasetConfig cfg = cfg_r.value();

    auto schema_r = slmkg::load_kg_schema(schema_path);
    if (schema_r.is_err()) { std::cerr << schema_r.error_message() << "\n"; return 1; }

    slmkg::GraphStore store;
    auto lr = store.load(kg_dir);
    if (lr.is_err()) { std::cerr << lr.error_message() << "\n"; return 1; }

    int written = 0, skipped = 0;

    if (kind == "pretrain") {
        std::vector<slmkg::ChunkRecord> corpus_chunks;
        std::string chunks_path = f.get("corpus-chunks", cfg.corpus_chunks_path);
        if (!chunks_path.empty()) {
            std::ifstream cf(chunks_path);
            if (!cf) {
                std::cerr << "Cannot open corpus-chunks: " << chunks_path << "\n"; return 1;
            }
            std::string line;
            while (std::getline(cf, line)) {
                if (line.empty()) continue;
                auto j = nlohmann::json::parse(line);
                slmkg::ChunkRecord ch;
                ch.id         = j.value("id", "");
                ch.document_id= j.value("document_id", "");
                ch.chunk_index= j.value("chunk_index", 0);
                ch.text       = j.value("text", "");
                ch.char_start = j.value("char_start", 0);
                ch.char_end   = j.value("char_end", 0);
                ch.token_count= j.value("token_count", 0);
                ch.text_hash  = j.value("text_hash", "");
                ch.split      = j.value("split", "train");
                corpus_chunks.push_back(std::move(ch));
            }
            std::cout << "Loaded " << corpus_chunks.size()
                      << " corpus chunks from " << chunks_path << "\n";
        }
        auto r = slmkg::build_pretrain_dataset(store, schema_r.value(), cfg, snapshot_id, output_path, corpus_chunks);
        written = r.written; skipped = r.skipped;
    } else if (kind == "sft") {
        auto r = slmkg::build_sft_dataset(store, schema_r.value(), cfg, snapshot_id, output_path);
        written = r.written; skipped = r.skipped;
    } else if (kind == "preference") {
        auto r = slmkg::build_preference_dataset(store, schema_r.value(), cfg, snapshot_id, output_path);
        written = r.written; skipped = r.skipped;
    } else {
        auto r = slmkg::build_eval_dataset(store, cfg, snapshot_id, output_path);
        written = r.written; skipped = r.skipped;
    }

    // Write manifest alongside output
    std::string manifest_path = fs::path(output_path).replace_extension(".manifest.json").string();
    std::string dataset_id = fs::path(output_path).stem().string();
    auto manifest = slmkg::compute_manifest(output_path, dataset_id, snapshot_id, cfg.generator);
    manifest.skipped_records = skipped;
    slmkg::write_manifest(manifest, manifest_path);

    std::cout << "Built " << kind << " dataset: " << written << " records"
              << (skipped > 0 ? " (" + std::to_string(skipped) + " skipped)" : "")
              << " -> " << output_path << "\n"
              << "Manifest: " << manifest_path << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// train download / train sft / eval run  (Phase 5 — Python bridge)
// ---------------------------------------------------------------------------
static bool venv_exists() {
    return fs::exists(".venv/bin/python") || fs::exists(".venv/Scripts/python.exe");
}

static int run_python(const std::string& cmd_str) {
    std::cout << "[slmkg] Running: " << cmd_str << "\n";
    int rc = std::system(cmd_str.c_str());
#ifdef _WIN32
    return rc;
#else
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : 1;
#endif
}

static int cmd_train_download(const Flags& f) {
    if (!venv_exists()) {
        std::cerr << "[train download] .venv not found. Run scripts/setup_env.sh first.\n";
        return 1;
    }
    std::string checkpoint = f.get("checkpoint", "openai-community/gpt2");
    std::string output     = f.get("output", "checkpoints/base/gpt2-small");
    std::string cmd_str =
        ".venv/bin/python scripts/download_checkpoint.py"
        " --checkpoint " + checkpoint +
        " --output "     + output;
    return run_python(cmd_str);
}

static int cmd_train_sft(const Flags& f) {
    if (!venv_exists()) {
        std::cerr << "[train sft] .venv not found. Run scripts/setup_env.sh first.\n";
        return 1;
    }
    std::string base   = f.get("base-checkpoint");
    std::string data   = f.get("dataset");
    std::string mcfg   = f.get("model-config",    "configs/model.yaml");
    std::string tcfg   = f.get("training-config", "configs/sft.yaml");
    std::string output = f.get("output");
    if (base.empty() || data.empty() || output.empty()) {
        std::cerr << "Usage: slmkg train sft "
                     "--base-checkpoint <path> "
                     "--dataset <path> "
                     "--output <path> "
                     "[--model-config configs/model.yaml] "
                     "[--training-config configs/sft.yaml]\n";
        return 1;
    }
    std::string cmd_str =
        ".venv/bin/python scripts/finetune_sft.py"
        " --base-checkpoint " + base   +
        " --dataset "         + data   +
        " --model-config "    + mcfg   +
        " --training-config " + tcfg   +
        " --output "          + output;
    return run_python(cmd_str);
}

static int cmd_eval_run(const Flags& f) {
    if (!venv_exists()) {
        std::cerr << "[eval run] .venv not found. Run scripts/setup_env.sh first.\n";
        return 1;
    }
    std::string checkpoint = f.get("checkpoint");
    std::string eval_data  = f.get("eval-dataset",
                                   "data/datasets/eval/eval_v0.1.jsonl");
    std::string output     = f.get("output");
    std::string max_tokens = f.get("max-new-tokens", "64");
    std::string tokenizer  = f.get("tokenizer", "");
    if (checkpoint.empty() || output.empty()) {
        std::cerr << "Usage: slmkg eval run "
                     "--checkpoint <path> "
                     "--output <path> "
                     "[--eval-dataset data/datasets/eval/eval_v0.1.jsonl] "
                     "[--tokenizer <path>] "
                     "[--max-new-tokens 64]\n";
        return 1;
    }
    std::string cmd_str =
        ".venv/bin/python scripts/evaluate_checkpoint.py"
        " --checkpoint "    + checkpoint +
        " --eval-dataset "  + eval_data  +
        " --output "        + output     +
        " --max-new-tokens "+ max_tokens;
    if (!tokenizer.empty())
        cmd_str += " --tokenizer " + tokenizer;
    return run_python(cmd_str);
}

// ---------------------------------------------------------------------------
// benchmark compare  (Phase 6 — Python bridge)
// ---------------------------------------------------------------------------
static int cmd_train_dpo(const Flags& f) {
    if (!venv_exists()) {
        std::cerr << "[train dpo] .venv not found. Run scripts/setup_env.sh first.\n";
        return 1;
    }
    std::string base   = f.get("base-checkpoint");
    std::string data   = f.get("preference-dataset");
    std::string mcfg   = f.get("model-config",    "configs/model.yaml");
    std::string tcfg   = f.get("training-config", "configs/dpo.yaml");
    std::string output = f.get("output");
    if (base.empty() || data.empty() || output.empty()) {
        std::cerr << "Usage: slmkg train dpo "
                     "--base-checkpoint <path> "
                     "--preference-dataset <path> "
                     "--output <path> "
                     "[--model-config configs/model.yaml] "
                     "[--training-config configs/dpo.yaml]\n";
        return 1;
    }
    std::string cmd_str =
        ".venv/bin/python scripts/finetune_dpo.py"
        " --base-checkpoint "    + base   +
        " --preference-dataset " + data   +
        " --model-config "       + mcfg   +
        " --training-config "    + tcfg   +
        " --output "             + output;
    return run_python(cmd_str);
}

static int cmd_eval_reward(const Flags& f) {
    if (!venv_exists()) {
        std::cerr << "[eval reward] .venv not found. Run scripts/setup_env.sh first.\n";
        return 1;
    }
    std::string checkpoint = f.get("checkpoint");
    std::string eval_data  = f.get("eval-dataset",
                                   "data/datasets/eval/eval_v0.1.jsonl");
    std::string output     = f.get("output");
    std::string max_tokens = f.get("max-new-tokens", "64");
    std::string tokenizer  = f.get("tokenizer", "");
    if (checkpoint.empty() || output.empty()) {
        std::cerr << "Usage: slmkg eval reward "
                     "--checkpoint <path> "
                     "--output <path> "
                     "[--eval-dataset data/datasets/eval/eval_v0.1.jsonl] "
                     "[--tokenizer <path>] "
                     "[--max-new-tokens 64]\n";
        return 1;
    }
    std::string cmd_str =
        ".venv/bin/python scripts/evaluate_kg_reward.py"
        " --checkpoint "     + checkpoint +
        " --eval-dataset "   + eval_data  +
        " --output "         + output     +
        " --max-new-tokens " + max_tokens;
    if (!tokenizer.empty()) cmd_str += " --tokenizer " + tokenizer;
    return run_python(cmd_str);
}

static int cmd_prompt_model(const Flags& f) {
    if (!venv_exists()) {
        std::cerr << "[prompt-model] .venv not found. Run scripts/setup_env.sh first.\n";
        return 1;
    }
    std::string checkpoint  = f.get("checkpoint");
    std::string compare     = f.get("compare", "");
    std::string temperature = f.get("temperature", "0.8");
    std::string top_p       = f.get("top-p", "0.95");
    std::string max_tokens  = f.get("max-new-tokens", "80");
    bool no_sample          = f.has("no-sample");
    if (checkpoint.empty()) {
        std::cerr << "Usage: slmkg prompt-model --checkpoint <path> "
                     "[--compare <path>] [--temperature 0.8] "
                     "[--top-p 0.95] [--max-new-tokens 80] [--no-sample]\n";
        return 1;
    }
    std::string cmd_str =
        ".venv/bin/python scripts/prompt.py"
        " --checkpoint "    + checkpoint   +
        " --temperature "   + temperature  +
        " --top-p "         + top_p        +
        " --max-new-tokens "+ max_tokens;
    if (!compare.empty()) cmd_str += " --compare " + compare;
    if (no_sample)        cmd_str += " --no-sample";
    return run_python(cmd_str);
}

static int cmd_benchmark_compare(int argc, char* argv[]) {
    if (!venv_exists()) {
        std::cerr << "[benchmark compare] .venv not found. Run scripts/setup_env.sh first.\n";
        return 1;
    }

    // Collect --runs <path>... and --output <path> by scanning argv directly,
    // because Flags only handles single-value arguments.
    std::vector<std::string> runs;
    std::string output;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--runs") {
            for (++i; i < argc && argv[i][0] != '-'; ++i)
                runs.push_back(argv[i]);
            --i;
        } else if (a == "--output" && i + 1 < argc) {
            output = argv[++i];
        }
    }

    if (runs.empty() || output.empty()) {
        std::cerr << "Usage: slmkg benchmark compare "
                     "--runs <report1.json> [report2.json ...] "
                     "--output <comparison.md>\n";
        return 1;
    }

    std::string cmd_str = ".venv/bin/python scripts/benchmark_compare.py --output " + output
                        + " --runs";
    for (const auto& r : runs) cmd_str += " " + r;
    return run_python(cmd_str);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: slmkg <command> <subcommand> [flags]\n"
                  << "  slmkg corpus ingest|validate|stats\n"
                  << "  slmkg prompt render|hash|validate\n"
                  << "  slmkg kg extract|normalize|validate|snapshot\n"
                  << "  slmkg dataset build-pretrain|build-sft|build-preference|build-eval\n"
                  << "  slmkg train download|sft|dpo\n"
                  << "  slmkg eval run|reward\n"
                  << "  slmkg benchmark compare\n"
                  << "  slmkg prompt-model --checkpoint <path> [--compare <path>]\n";
        return 1;
    }

    std::string cmd    = argv[1];
    // prompt-model is a single-keyword command; parse flags from argv[2].
    if (cmd == "prompt-model") {
        Flags flags = Flags::parse(argc, argv, 2);
        return cmd_prompt_model(flags);
    }

    if (argc < 3) {
        std::cerr << "Usage: slmkg <command> <subcommand> [flags]\n"; return 1;
    }
    std::string subcmd = argv[2];
    Flags flags = Flags::parse(argc, argv, 3);

    if      (cmd == "corpus" && subcmd == "ingest")   return cmd_corpus_ingest(flags);
    else if (cmd == "corpus" && subcmd == "validate")  return cmd_corpus_validate(flags);
    else if (cmd == "corpus" && subcmd == "stats")     return cmd_corpus_stats(flags);
    else if (cmd == "prompt" && subcmd == "render")    return cmd_prompt_render(flags);
    else if (cmd == "prompt" && subcmd == "hash")      return cmd_prompt_hash(flags);
    else if (cmd == "prompt" && subcmd == "validate")  return cmd_prompt_validate(flags);
    else if (cmd == "kg"     && subcmd == "extract")   return cmd_kg_extract(flags);
    else if (cmd == "kg"     && subcmd == "normalize") return cmd_kg_normalize(flags);
    else if (cmd == "kg"     && subcmd == "validate")  return cmd_kg_validate(flags);
    else if (cmd == "kg"     && subcmd == "snapshot")  return cmd_kg_snapshot(flags);
    else if (cmd == "dataset" && subcmd == "build-pretrain")  return cmd_dataset_build(flags, "pretrain");
    else if (cmd == "dataset" && subcmd == "build-sft")       return cmd_dataset_build(flags, "sft");
    else if (cmd == "dataset" && subcmd == "build-preference") return cmd_dataset_build(flags, "preference");
    else if (cmd == "dataset" && subcmd == "build-eval")      return cmd_dataset_build(flags, "eval");
    else if (cmd == "train"     && subcmd == "download") return cmd_train_download(flags);
    else if (cmd == "train"     && subcmd == "sft")      return cmd_train_sft(flags);
    else if (cmd == "train"     && subcmd == "dpo")      return cmd_train_dpo(flags);
    else if (cmd == "eval"      && subcmd == "run")      return cmd_eval_run(flags);
    else if (cmd == "eval"      && subcmd == "reward")   return cmd_eval_reward(flags);
    else if (cmd == "benchmark" && subcmd == "compare")  return cmd_benchmark_compare(argc, argv);
    else {
        std::cerr << "Unknown command: " << cmd << " " << subcmd << "\n";
        return 1;
    }
}
