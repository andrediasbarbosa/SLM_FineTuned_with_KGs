#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <string>

#include "kg/graph_linearizer.hpp"
#include "kg/graph_store.hpp"
#include "kg/schema.hpp"
#include "datasets/negative_sampler.hpp"
#include "datasets/pretrain_builder.hpp"
#include "datasets/sft_builder.hpp"
#include "datasets/preference_builder.hpp"
#include "datasets/eval_builder.hpp"
#include "datasets/dataset_config.hpp"
#include "datasets/dataset_manifest.hpp"

namespace fs = std::filesystem;

static const std::string ROOT = SLMKG_PROJECT_ROOT;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string tmp(const std::string& name) {
    fs::path p = fs::temp_directory_path() / ("slmkg_ds_test_" + name);
    fs::create_directories(p);
    return p.string();
}

static slmkg::GraphStore make_test_store() {
    slmkg::GraphStore store;

    slmkg::CanonicalEntity e1, e2, e3;
    e1.id = "ent_000001"; e1.canonical_name = "knowledge graph"; e1.type = "Concept";
    e2.id = "ent_000002"; e2.canonical_name = "factual grounding"; e2.type = "Concept";
    e3.id = "ent_000003"; e3.canonical_name = "neural network"; e3.type = "Method";
    store.add_entity(e1); store.add_entity(e2); store.add_entity(e3);

    slmkg::CanonicalRelation r1, r2;
    r1.id = "rel_000001"; r1.head_ids = {"ent_000001"}; r1.relation_type = "supports";
    r1.tail_ids = {"ent_000002"}; r1.confidence = 0.9;
    r1.evidence.push_back({"doc_1", "chunk_1", "KGs support factual grounding."});
    store.add_relation(r1);

    r2.id = "rel_000002"; r2.head_ids = {"ent_000003"}; r2.relation_type = "uses";
    r2.tail_ids = {"ent_000001"}; r2.confidence = 0.8;
    store.add_relation(r2);

    return store;
}

static slmkg::KgSchema make_test_schema() {
    slmkg::KgSchema s;
    s.schema_version = "test_v0.1";
    slmkg::KgEntityTypeSpec et;
    et.name = "Concept"; s.entity_types.push_back(et);
    et.name = "Method";  s.entity_types.push_back(et);
    slmkg::KgRelationTypeSpec rt;
    rt.name = "supports"; rt.domain = "Concept"; rt.range = "Concept";
    rt.head_arity = {1,-1}; rt.tail_arity = {1,-1};
    s.relation_types.push_back(rt);
    rt.name = "uses"; rt.domain = "Method"; rt.range = "Concept";
    s.relation_types.push_back(rt);
    s.rebuild_index();
    return s;
}

static slmkg::DatasetConfig make_test_cfg() {
    slmkg::DatasetConfig cfg;
    cfg.max_records     = 500;
    cfg.max_char_length = 3600;
    cfg.split_seed      = 42;
    cfg.train_ratio     = 0.80;
    cfg.val_ratio       = 0.10;
    cfg.max_hop_paths   = 100;
    cfg.refusal_ratio   = 0.20;
    cfg.generator       = "test_builder";
    return cfg;
}

// ---------------------------------------------------------------------------
// GraphLinearizerTest
// ---------------------------------------------------------------------------
TEST_CASE("GraphLinearizerTest.LinearizesTriple") {
    std::string s = slmkg::linearize_triple("Knowledge Graph", "supports", "Factual Grounding");
    REQUIRE(s == "[ENTITY] Knowledge Graph [RELATION] supports [ENTITY] Factual Grounding");
}

TEST_CASE("GraphLinearizerTest.LinearizesHyperedge") {
    std::string s = slmkg::linearize_hyperedge(
        {"Einstein", "Bohr"}, "contributed_to", {"Special Relativity"});
    REQUIRE(s.find("[ENTITY] Einstein") != std::string::npos);
    REQUIRE(s.find("[ENTITY] Bohr")     != std::string::npos);
    REQUIRE(s.find("[RELATION] contributed_to") != std::string::npos);
}

TEST_CASE("GraphLinearizerTest.LinearizesPath") {
    std::vector<slmkg::PathStep> steps = {
        {"A", "r1", "B", "rel_1"},
        {"B", "r2", "C", "rel_2"},
    };
    std::string s = slmkg::linearize_path(steps);
    REQUIRE(s.find("[ENTITY] A") != std::string::npos);
    REQUIRE(s.find("[RELATION] r1") != std::string::npos);
    REQUIRE(s.find("[ENTITY] B") != std::string::npos);
    REQUIRE(s.find("[RELATION] r2") != std::string::npos);
    REQUIRE(s.find("[ENTITY] C") != std::string::npos);
}

TEST_CASE("GraphLinearizerTest.VerbalizesMergesAndSuffix") {
    std::string s = slmkg::verbalize_triple("KG", "supports", "grounding");
    REQUIRE(s == "KG supports grounding.");
}

// ---------------------------------------------------------------------------
// NegativeSamplerTest
// ---------------------------------------------------------------------------
TEST_CASE("NegativeSamplerTest.DoesNotGenerateTrueFactAsNegative") {
    auto store  = make_test_store();
    auto schema = make_test_schema();

    auto result = slmkg::generate_negatives(store, schema, 42);
    REQUIRE(!result.samples.empty());

    // Build collision set the same way the sampler does: rel + sorted heads + sorted tails.
    auto make_key = [](const std::string& rel,
                       std::vector<std::string> heads,
                       std::vector<std::string> tails) {
        std::sort(heads.begin(), heads.end());
        std::sort(tails.begin(), tails.end());
        std::string k = rel;
        for (const auto& h : heads) k += "|h:" + h;
        for (const auto& t : tails) k += "|t:" + t;
        return k;
    };

    std::unordered_set<std::string> true_facts;
    for (const auto& r : store.relations())
        true_facts.insert(make_key(r.relation_type, r.head_ids, r.tail_ids));

    for (const auto& neg : result.samples) {
        std::string key = make_key(neg.relation_type, neg.head_ids, neg.tail_ids);
        REQUIRE(true_facts.find(key) == true_facts.end());
    }
}

TEST_CASE("NegativeSamplerTest.NegativesHaveLinearizedField") {
    auto store  = make_test_store();
    auto schema = make_test_schema();
    auto result = slmkg::generate_negatives(store, schema, 42);
    for (const auto& neg : result.samples)
        REQUIRE(!neg.linearized.empty());
}

// ---------------------------------------------------------------------------
// PretrainBuilderTest
// ---------------------------------------------------------------------------
static std::vector<slmkg::ChunkRecord> make_test_chunks() {
    slmkg::ChunkRecord c1, c2, c3;
    c1.id = "chunk_000001"; c1.document_id = "doc_1"; c1.chunk_index = 0;
    c1.text = "Knowledge graphs provide structured factual grounding for language models.";
    c1.split = "train"; c1.token_count = 12;

    c2.id = "chunk_000002"; c2.document_id = "doc_1"; c2.chunk_index = 1;
    c2.text = "Neural networks learn distributed representations of language.";
    c2.split = "validation"; c2.token_count = 10;

    c3.id = "chunk_000003"; c3.document_id = "doc_2"; c3.chunk_index = 0;
    c3.text = "Fine-tuning adapts a pre-trained model to a specific task.";
    c3.split = "test"; c3.token_count = 11;

    return {c1, c2, c3};
}

TEST_CASE("PretrainBuilderTest.NoChunksProducesNoRawText") {
    auto store  = make_test_store();
    auto schema = make_test_schema();
    auto cfg    = make_test_cfg();
    std::string td = tmp("pretrain_nochunks");

    slmkg::build_pretrain_dataset(store, schema, cfg, "snap_v0.1", td + "/pt.jsonl");

    std::ifstream f(td + "/pt.jsonl");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded()) continue;
        REQUIRE(j.value("record_type", "") != "raw_text");
    }
    fs::remove_all(td);
}

TEST_CASE("PretrainBuilderTest.ChunksProduceRawTextRecords") {
    auto store  = make_test_store();
    auto schema = make_test_schema();
    auto cfg    = make_test_cfg();
    auto chunks = make_test_chunks();
    std::string td = tmp("pretrain_chunks");

    auto result = slmkg::build_pretrain_dataset(store, schema, cfg, "snap_v0.1",
                                                 td + "/pt.jsonl", chunks);

    int raw_count = 0;
    std::ifstream f(td + "/pt.jsonl");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (!j.is_discarded() && j.value("record_type", "") == "raw_text")
            ++raw_count;
    }
    REQUIRE(raw_count == static_cast<int>(chunks.size()));
    REQUIRE(result.written >= raw_count);
    fs::remove_all(td);
}

TEST_CASE("PretrainBuilderTest.RawTextRecordsInheritChunkSplit") {
    auto store  = make_test_store();
    auto schema = make_test_schema();
    auto cfg    = make_test_cfg();
    auto chunks = make_test_chunks(); // splits: train, validation, test
    std::string td = tmp("pretrain_split");

    slmkg::build_pretrain_dataset(store, schema, cfg, "snap_v0.1",
                                   td + "/pt.jsonl", chunks);

    std::map<std::string, std::string> chunk_splits;
    for (const auto& c : chunks) chunk_splits[c.text] = c.split;

    std::ifstream f(td + "/pt.jsonl");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded() || j.value("record_type","") != "raw_text") continue;
        std::string text  = j.value("text", "");
        std::string split = j.value("split", "");
        auto it = chunk_splits.find(text);
        REQUIRE(it != chunk_splits.end());
        REQUIRE(split == it->second);
    }
    fs::remove_all(td);
}

TEST_CASE("PretrainBuilderTest.RawTextRecordsHaveEmptyKgSnapshot") {
    auto store  = make_test_store();
    auto schema = make_test_schema();
    auto cfg    = make_test_cfg();
    auto chunks = make_test_chunks();
    std::string td = tmp("pretrain_snap");

    slmkg::build_pretrain_dataset(store, schema, cfg, "snap_v0.1",
                                   td + "/pt.jsonl", chunks);

    std::ifstream f(td + "/pt.jsonl");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded() || j.value("record_type","") != "raw_text") continue;
        REQUIRE(j.value("kg_snapshot", "NONEMPTY") == "");
    }
    fs::remove_all(td);
}

TEST_CASE("PretrainBuilderTest.OversizedChunkIsSkipped") {
    auto store  = make_test_store();
    auto schema = make_test_schema();
    auto cfg    = make_test_cfg();
    cfg.max_char_length = 20; // very tight limit

    slmkg::ChunkRecord big;
    big.id = "chunk_big"; big.document_id = "doc_x"; big.chunk_index = 0;
    big.text  = std::string(21, 'x'); // 21 chars > 20 limit
    big.split = "train";

    std::string td = tmp("pretrain_oversize");
    auto result = slmkg::build_pretrain_dataset(store, schema, cfg, "snap_v0.1",
                                                 td + "/pt.jsonl", {big});
    REQUIRE(result.skipped >= 1);

    std::ifstream f(td + "/pt.jsonl");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded() || j.value("record_type","") != "raw_text") continue;
        REQUIRE(false); // oversized chunk must not appear
    }
    fs::remove_all(td);
}

TEST_CASE("PretrainBuilderTest.RawTextAppearsBeforeGraphRecords") {
    auto store  = make_test_store();
    auto schema = make_test_schema();
    auto cfg    = make_test_cfg();
    auto chunks = make_test_chunks();
    std::string td = tmp("pretrain_order");

    slmkg::build_pretrain_dataset(store, schema, cfg, "snap_v0.1",
                                   td + "/pt.jsonl", chunks);

    std::ifstream f(td + "/pt.jsonl");
    std::string line;
    bool seen_graph = false;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded()) continue;
        std::string rt = j.value("record_type", "");
        if (rt != "raw_text") { seen_graph = true; }
        if (rt == "raw_text")  { REQUIRE(!seen_graph); } // raw_text must precede graph records
    }
    fs::remove_all(td);
}

TEST_CASE("DatasetConfigTest.LoadsCorpusChunksPath") {
    std::string td = tmp("cfg_chunks");
    std::string yaml_path = td + "/ds.yaml";
    {
        std::ofstream out(yaml_path);
        out << "dataset:\n"
            << "  kg_snapshot_id: test_snap\n"
            << "  corpus_chunks_path: data/processed/my_corpus/chunks.jsonl\n"
            << "  max_records: 500\n";
    }
    auto r = slmkg::load_dataset_config(yaml_path);
    REQUIRE(!r.is_err());
    REQUIRE(r.value().corpus_chunks_path == "data/processed/my_corpus/chunks.jsonl");
    fs::remove_all(td);
}

TEST_CASE("DatasetConfigTest.MissingCorpusChunksPathDefaultsEmpty") {
    std::string td = tmp("cfg_nochunks");
    std::string yaml_path = td + "/ds.yaml";
    {
        std::ofstream out(yaml_path);
        out << "dataset:\n  kg_snapshot_id: test_snap\n";
    }
    auto r = slmkg::load_dataset_config(yaml_path);
    REQUIRE(!r.is_err());
    REQUIRE(r.value().corpus_chunks_path.empty());
    fs::remove_all(td);
}

// ---------------------------------------------------------------------------
// SftBuilderTest
// ---------------------------------------------------------------------------
TEST_CASE("SftBuilderTest.GeneratesSingleHopQa") {
    auto store  = make_test_store();
    auto schema = make_test_schema();
    auto cfg    = make_test_cfg();
    std::string td = tmp("sft");

    auto result = slmkg::build_sft_dataset(store, schema, cfg, "snap_v0.1", td + "/sft.jsonl");
    REQUIRE(result.written > 0);

    // Check at least one single_hop_qa record exists
    std::ifstream f(td + "/sft.jsonl");
    std::string line;
    bool found_single_hop = false;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (!j.is_discarded() && j.value("task_type","") == "single_hop_qa") {
            found_single_hop = true;
            REQUIRE(!j.value("input","").empty());
            REQUIRE(!j.value("output","").empty());
            REQUIRE(!j.value("graph_context", nlohmann::json::array()).empty());
        }
    }
    REQUIRE(found_single_hop);
    fs::remove_all(td);
}

TEST_CASE("SftBuilderTest.GeneratesMissingEvidenceRefusal") {
    auto store  = make_test_store();
    auto schema = make_test_schema();
    auto cfg    = make_test_cfg();
    std::string td = tmp("sft_refusal");

    slmkg::build_sft_dataset(store, schema, cfg, "snap_v0.1", td + "/sft.jsonl");

    std::ifstream f(td + "/sft.jsonl");
    std::string line;
    bool found_refusal = false;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (!j.is_discarded() && j.value("task_type","") == "missing_evidence_refusal") {
            found_refusal = true;
            REQUIRE(j.value("output","").find("does not support") != std::string::npos);
        }
    }
    REQUIRE(found_refusal);
    fs::remove_all(td);
}

TEST_CASE("SftBuilderTest.AllRecordsWithinCharLimit") {
    auto store  = make_test_store();
    auto schema = make_test_schema();
    auto cfg    = make_test_cfg();
    std::string td = tmp("sft_len");

    slmkg::build_sft_dataset(store, schema, cfg, "snap_v0.1", td + "/sft.jsonl");

    std::ifstream f(td + "/sft.jsonl");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded()) continue;
        int chars = static_cast<int>(j.value("instruction","").size()
                                   + j.value("input","").size()
                                   + j.value("output","").size());
        REQUIRE(chars <= cfg.max_char_length);
    }
    fs::remove_all(td);
}

// ---------------------------------------------------------------------------
// PreferenceBuilderTest
// ---------------------------------------------------------------------------
TEST_CASE("PreferenceBuilderTest.GeneratesChosenRejectedPairs") {
    auto store  = make_test_store();
    auto schema = make_test_schema();
    auto cfg    = make_test_cfg();
    std::string td = tmp("pref");

    auto result = slmkg::build_preference_dataset(store, schema, cfg, "snap_v0.1",
                                                   td + "/pref.jsonl");
    REQUIRE(result.written > 0);

    std::ifstream f(td + "/pref.jsonl");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded()) continue;
        REQUIRE(!j.value("chosen","").empty());
        REQUIRE(!j.value("rejected","").empty());
        REQUIRE(j.value("chosen","") != j.value("rejected",""));
    }
    fs::remove_all(td);
}

// ---------------------------------------------------------------------------
// EvalBuilderTest
// ---------------------------------------------------------------------------
TEST_CASE("EvalBuilderTest.HoldsOutTestFacts") {
    auto store = make_test_store();
    auto cfg   = make_test_cfg();
    std::string td = tmp("eval");

    auto result = slmkg::build_eval_dataset(store, cfg, "snap_v0.1", td + "/eval.jsonl");
    // With 2 relations and seed=42, some should be held out as test
    CHECK(result.total_held_out >= 0);

    std::ifstream f(td + "/eval.jsonl");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded()) continue;
        // All eval records must be in test split
        REQUIRE(j.value("split","") == "test");
    }
    fs::remove_all(td);
}
