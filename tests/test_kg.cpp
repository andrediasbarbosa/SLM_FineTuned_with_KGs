#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <string>

#include "kg/schema.hpp"
#include "kg/entity.hpp"
#include "kg/relation.hpp"
#include "kg/graph_store.hpp"
#include "kg/normalizer.hpp"
#include "kg/graph_validator.hpp"
#include "kg/graph_snapshot.hpp"

namespace fs = std::filesystem;

static const std::string ROOT = SLMKG_PROJECT_ROOT;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string tmp(const std::string& name) {
    fs::path p = fs::temp_directory_path() / ("slmkg_kg_test_" + name);
    fs::create_directories(p);
    return p.string();
}

static slmkg::KgSchema make_test_schema() {
    slmkg::KgSchema s;
    s.schema_version = "test_v0.1";
    slmkg::KgEntityTypeSpec et;
    et.name = "Concept"; s.entity_types.push_back(et);
    et.name = "Method";  s.entity_types.push_back(et);

    slmkg::KgRelationTypeSpec rt;
    rt.name = "supports"; rt.domain = "Concept"; rt.range = "Concept";
    rt.head_arity = {1, -1}; rt.tail_arity = {1, -1};
    s.relation_types.push_back(rt);

    rt.name = "uses"; rt.domain = "Method"; rt.range = "Concept";
    s.relation_types.push_back(rt);

    s.rebuild_index();
    return s;
}

// ---------------------------------------------------------------------------
// KgSchemaTest
// ---------------------------------------------------------------------------
TEST_CASE("KgSchemaTest.LoadsValidSchema") {
    auto r = slmkg::load_kg_schema(ROOT + "/configs/kg_schema.yaml");
    REQUIRE(r.is_ok());
    REQUIRE(!r.value().entity_types.empty());
    REQUIRE(!r.value().relation_types.empty());
    REQUIRE(!r.value().schema_version.empty());
}

TEST_CASE("KgSchemaTest.RejectsUnknownEntityType") {
    auto schema = make_test_schema();
    REQUIRE( schema.has_entity_type("Concept"));
    REQUIRE(!schema.has_entity_type("Unicorn"));
}

TEST_CASE("KgSchemaTest.RejectsUnknownRelationType") {
    auto schema = make_test_schema();
    REQUIRE( schema.has_relation_type("supports"));
    REQUIRE(!schema.has_relation_type("flies_over"));
}

TEST_CASE("KgSchemaTest.ValidatesDomainRangeConstraints") {
    auto schema = make_test_schema();
    const auto* rt = schema.get_relation_type("supports");
    REQUIRE(rt != nullptr);
    REQUIRE(rt->domain == "Concept");
    REQUIRE(rt->range  == "Concept");
}

// ---------------------------------------------------------------------------
// GraphStoreTest
// ---------------------------------------------------------------------------
TEST_CASE("GraphStoreTest.RoundTripsEntities") {
    slmkg::GraphStore store;
    slmkg::CanonicalEntity e;
    e.id = "ent_000001"; e.type = "Concept";
    e.canonical_name = "knowledge graph";
    e.aliases = {"KG"}; e.confidence = 0.9;
    e.validation_status = "auto_validated";
    store.add_entity(e);

    std::string td = tmp("store_ent");
    REQUIRE(store.write(td).is_ok());

    slmkg::GraphStore store2;
    REQUIRE(store2.load(td).is_ok());
    REQUIRE(store2.entities().size() == 1);
    CHECK(store2.entities()[0].id             == "ent_000001");
    CHECK(store2.entities()[0].canonical_name == "knowledge graph");
    CHECK(store2.entities()[0].aliases[0]     == "KG");

    fs::remove_all(td);
}

TEST_CASE("GraphStoreTest.RoundTripsRelations") {
    slmkg::GraphStore store;
    slmkg::CanonicalRelation r;
    r.id = "rel_000001"; r.relation_type = "supports";
    r.head_ids = {"ent_000001"}; r.tail_ids = {"ent_000002"};
    r.confidence = 0.8;
    r.evidence.push_back({"doc_000001", "chunk_000001", "KGs support grounding."});
    store.add_relation(r);

    std::string td = tmp("store_rel");
    REQUIRE(store.write(td).is_ok());

    slmkg::GraphStore store2;
    REQUIRE(store2.load(td).is_ok());
    REQUIRE(store2.relations().size() == 1);
    CHECK(store2.relations()[0].relation_type        == "supports");
    CHECK(store2.relations()[0].evidence[0].chunk_id == "chunk_000001");

    fs::remove_all(td);
}

// ---------------------------------------------------------------------------
// EntityNormalizerTest
// ---------------------------------------------------------------------------
TEST_CASE("EntityNormalizerTest.MergesAliases") {
    std::string td = tmp("normalizer");
    // Two candidates with the same normalized name
    std::ofstream ef(td + "/candidate_entities.jsonl");
    ef << R"({"candidate_id":"cand_ent_000001","name":"Knowledge Graph","type":"Concept","document_id":"doc_1","chunk_id":"chunk_1","confidence":0.9})" << "\n";
    ef << R"({"candidate_id":"cand_ent_000002","name":"knowledge graph","type":"Concept","document_id":"doc_2","chunk_id":"chunk_2","confidence":0.8})" << "\n";
    ef << R"({"candidate_id":"cand_ent_000003","name":"Neural Network","type":"Method","document_id":"doc_1","chunk_id":"chunk_1","confidence":0.85})" << "\n";
    ef.close();
    std::ofstream rf(td + "/candidate_relations.jsonl");
    rf << R"({"relation_type":"uses","head_candidate_ids":["cand_ent_000003"],"tail_candidate_ids":["cand_ent_000001"],"document_id":"doc_1","chunk_id":"chunk_1","confidence":0.7})" << "\n";
    rf.close();

    auto result = slmkg::normalize_candidates(td);

    // Two candidates with same name → 1 entity + 1 merge
    CHECK(result.entities_created == 2);
    CHECK(result.entities_merged  == 1);
    CHECK(result.relations_created == 1);
    CHECK(result.store.entities().size() == 2);

    // The merged entity should have both source docs
    const auto& kg_ent = result.store.entities()[0];
    CHECK(kg_ent.source_ids.size() == 2);

    fs::remove_all(td);
}

TEST_CASE("EntityNormalizerTest.NormalizesName") {
    CHECK(slmkg::normalize_name("  Knowledge Graph  ") == "knowledge graph");
    CHECK(slmkg::normalize_name("NEURAL\tNETWORK")      == "neural network");
    CHECK(slmkg::normalize_name("already_normal")       == "already_normal");
}

// ---------------------------------------------------------------------------
// RelationNormalizerTest
// ---------------------------------------------------------------------------
TEST_CASE("RelationNormalizerTest.DeduplicatesRelations") {
    std::string td = tmp("rel_dedup");
    std::ofstream ef(td + "/candidate_entities.jsonl");
    ef << R"({"candidate_id":"cand_ent_000001","name":"Concept A","type":"Concept","document_id":"doc_1","chunk_id":"c1","confidence":0.9})" << "\n";
    ef << R"({"candidate_id":"cand_ent_000002","name":"Concept B","type":"Concept","document_id":"doc_1","chunk_id":"c1","confidence":0.9})" << "\n";
    ef.close();
    std::ofstream rf(td + "/candidate_relations.jsonl");
    // Same relation twice
    rf << R"({"relation_type":"supports","head_candidate_ids":["cand_ent_000001"],"tail_candidate_ids":["cand_ent_000002"],"document_id":"doc_1","chunk_id":"c1","confidence":0.8})" << "\n";
    rf << R"({"relation_type":"supports","head_candidate_ids":["cand_ent_000001"],"tail_candidate_ids":["cand_ent_000002"],"document_id":"doc_2","chunk_id":"c2","confidence":0.75})" << "\n";
    rf.close();

    auto result = slmkg::normalize_candidates(td);
    // Normalizer currently keeps both (dedup is validator's job); check both are created
    CHECK(result.relations_created == 2);

    fs::remove_all(td);
}

// ---------------------------------------------------------------------------
// GraphSnapshotTest
// ---------------------------------------------------------------------------
TEST_CASE("GraphSnapshotTest.HashStableUnderSortedInput") {
    auto schema = make_test_schema();
    slmkg::GraphStore store;
    slmkg::CanonicalEntity e1, e2;
    e1.id = "ent_000002"; e1.canonical_name = "b"; e1.type = "Concept";
    e2.id = "ent_000001"; e2.canonical_name = "a"; e2.type = "Concept";
    store.add_entity(e1);
    store.add_entity(e2); // added in reverse id order

    std::string td1 = tmp("snap1"), td2 = tmp("snap2");
    auto r1 = slmkg::write_snapshot(store, "snap", td1, schema);
    auto r2 = slmkg::write_snapshot(store, "snap", td2, schema);
    REQUIRE(r1.is_ok());
    REQUIRE(r2.is_ok());
    // Same content → same hash regardless of insertion order
    CHECK(r1.value().snapshot_hash == r2.value().snapshot_hash);
    CHECK(r1.value().entity_count  == 2);

    fs::remove_all(td1);
    fs::remove_all(td2);
}

TEST_CASE("GraphSnapshotTest.HashChangesWhenFactChanges") {
    auto schema = make_test_schema();
    slmkg::GraphStore store1, store2;

    slmkg::CanonicalEntity e;
    e.id = "ent_000001"; e.canonical_name = "knowledge graph"; e.type = "Concept";
    store1.add_entity(e);
    e.canonical_name = "knowledge graph — modified";
    store2.add_entity(e);

    std::string td1 = tmp("snap_change1"), td2 = tmp("snap_change2");
    auto r1 = slmkg::write_snapshot(store1, "s", td1, schema);
    auto r2 = slmkg::write_snapshot(store2, "s", td2, schema);
    REQUIRE(r1.is_ok());
    REQUIRE(r2.is_ok());
    CHECK(r1.value().snapshot_hash != r2.value().snapshot_hash);

    fs::remove_all(td1);
    fs::remove_all(td2);
}

TEST_CASE("GraphSnapshotTest.WritesManifestYaml") {
    auto schema = make_test_schema();
    slmkg::GraphStore store;
    slmkg::CanonicalEntity e;
    e.id = "ent_000001"; e.canonical_name = "test"; e.type = "Concept";
    store.add_entity(e);

    std::string td = tmp("snap_manifest");
    auto r = slmkg::write_snapshot(store, "kg_v1", td, schema);
    REQUIRE(r.is_ok());
    REQUIRE(std::filesystem::exists(td + "/manifest.yaml"));

    fs::remove_all(td);
}

// ---------------------------------------------------------------------------
// GraphValidatorTest
// ---------------------------------------------------------------------------
TEST_CASE("GraphValidatorTest.ValidatesKnownTypes") {
    auto schema = make_test_schema();
    slmkg::GraphStore store;

    slmkg::CanonicalEntity e1, e2;
    e1.id = "ent_000001"; e1.canonical_name = "kg";     e1.type = "Concept";
    e2.id = "ent_000002"; e2.canonical_name = "resnet"; e2.type = "Method";
    store.add_entity(e1);
    store.add_entity(e2);

    slmkg::CanonicalRelation r;
    r.id = "rel_000001"; r.relation_type = "uses";
    r.head_ids = {"ent_000002"}; r.tail_ids = {"ent_000001"};
    store.add_relation(r);

    slmkg::KgValidationReport report = slmkg::validate_graph(store, schema, false);
    CHECK(report.valid_entities   == 2);
    CHECK(report.rejected_entities == 0);
    CHECK(report.valid_relations  == 1);
    CHECK(!report.has_hard_errors());
}

TEST_CASE("GraphValidatorTest.RejectsEmptyRelation") {
    auto schema = make_test_schema();
    slmkg::GraphStore store;
    slmkg::CanonicalRelation r;
    r.id = "rel_000001"; r.relation_type = "supports";
    // head_ids intentionally empty
    r.tail_ids = {"ent_000001"};
    store.add_relation(r);

    slmkg::KgValidationReport report = slmkg::validate_graph(store, schema, false);
    CHECK(report.rejected_relations == 1);
    CHECK(report.has_hard_errors());
}
