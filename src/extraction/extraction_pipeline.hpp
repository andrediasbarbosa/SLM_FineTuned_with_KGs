#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <nlohmann/json.hpp>
#include "provider_interface.hpp"
#include "response_parser.hpp"
#include "extraction_cache.hpp"
#include "extraction_config.hpp"
#include "core/hashing.hpp"
#include "corpus/document.hpp"
#include "prompts/prompt_template.hpp"
#include "prompts/prompt_renderer.hpp"

namespace slmkg {
namespace fs = std::filesystem;

// Candidate records written to output JSONL (Sections 8.3, 8.4).
struct CandidateEntityRecord {
    std::string              candidate_id;
    std::string              document_id;
    std::string              chunk_id;
    std::string              name;
    std::string              type;
    std::vector<std::string> aliases;
    std::string              description;
    std::string              evidence_text;
    double                   confidence        = 0.0;
    std::string              provider;
    std::string              model;
    std::string              prompt_set_id;
    std::string              validation_status = "unvalidated";
};

struct CandidateRelationRecord {
    std::string              candidate_id;
    std::string              document_id;
    std::string              chunk_id;
    std::vector<std::string> head_candidate_ids;
    std::string              relation_type;
    std::vector<std::string> tail_candidate_ids;
    std::string              evidence_text;
    double                   confidence        = 0.0;
    std::string              provider;
    std::string              model;
    std::string              prompt_set_id;
    std::string              validation_status = "unvalidated";
};

struct PipelineResult {
    int chunks_processed  = 0;
    int chunks_skipped    = 0;   // served from cache
    int entities_written  = 0;
    int relations_written = 0;
    int rejected          = 0;
    std::vector<std::string> errors;
};

namespace detail {

inline std::string now_iso8601_utc() {
    std::time_t t = std::time(nullptr);
    std::tm* tm_utc = std::gmtime(&t);
    std::ostringstream oss;
    oss << std::put_time(tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

inline std::string make_request_id(int n) {
    std::ostringstream oss;
    oss << "extract_req_" << std::setw(6) << std::setfill('0') << n;
    return oss.str();
}

inline std::string make_candidate_id(const std::string& prefix, int n) {
    std::ostringstream oss;
    oss << prefix << std::setw(6) << std::setfill('0') << n;
    return oss.str();
}

inline nlohmann::json call_record_to_json(
        const ExtractionRequestMetadata& meta,
        const RenderedPrompt&            prompt,
        const ExtractionResponse&        resp,
        const std::string&               req_ts,
        const std::string&               resp_ts,
        double                           temperature,
        int                              max_tokens)
{
    return {
        {"request_id",           meta.request_id},
        {"provider",             resp.provider},
        {"model",                resp.model},
        {"document_id",          meta.document_id},
        {"chunk_id",             meta.chunk_id},
        {"prompt_set_id",        prompt.prompt_set_id},
        {"system_prompt_hash",   prompt.system_hash},
        {"user_prompt_hash",     prompt.user_hash},
        {"rendered_prompt_hash", prompt.rendered_hash},
        {"request_timestamp",    req_ts},
        {"response_timestamp",   resp_ts},
        {"temperature",          temperature},
        {"max_output_tokens",    max_tokens},
        {"status",               to_string(resp.status)},
        {"raw_response_hash",    resp.raw_response_hash},
    };
}

} // namespace detail

struct ExtractionPipelineConfig {
    KgExtractionConfig         cfg;
    ProviderConfig             provider_cfg;
    std::unordered_set<std::string> allowed_entity_types;
    std::unordered_set<std::string> allowed_relation_types;
    std::string                output_dir;
    std::string                record_to_dir; // if non-empty, also write raw responses here
};

inline PipelineResult run_extraction_pipeline(
        const std::vector<ChunkRecord>&  chunks,
        const PromptTemplate&            sys_tmpl,
        const PromptTemplate&            user_tmpl,
        IKgExtractionProvider&           provider,
        ExtractionCache&                 cache,
        const ExtractionPipelineConfig&  pcfg)
{
    PipelineResult result;

    fs::create_directories(pcfg.output_dir);

    std::ofstream f_entities  (pcfg.output_dir + "/candidate_entities.jsonl",  std::ios::app);
    std::ofstream f_relations (pcfg.output_dir + "/candidate_relations.jsonl", std::ios::app);
    std::ofstream f_call_recs (pcfg.output_dir + "/call_records.jsonl",        std::ios::app);
    std::ofstream f_rejected  (pcfg.output_dir + "/rejected_candidates.jsonl", std::ios::app);

    const auto& pset_id = pcfg.cfg.prompts.prompt_set_id;
    int req_counter = 1;
    int ent_counter = 1;
    int rel_counter = 1;

    for (const auto& chunk : chunks) {
        // Render the user prompt for this chunk.
        std::unordered_map<std::string, std::string> vars = {
            {"document_id",            chunk.document_id},
            {"chunk_id",               chunk.id},
            {"chunk_text",             chunk.text},
            {"kg_schema",              ""},
            {"allowed_entity_types",   ""},
            {"allowed_relation_types", ""},
            {"known_entities",         ""},
            {"extraction_mode",        pcfg.cfg.extraction_mode},
            {"output_schema",          ""},
            {"max_triples",            "50"},
        };
        auto rendered_r = render_prompt(user_tmpl, vars);
        if (rendered_r.is_err()) {
            result.errors.push_back("Chunk " + chunk.id + ": " + rendered_r.error_message());
            continue;
        }

        RenderedPrompt rp;
        rp.system_text    = sys_tmpl.content;
        rp.user_text      = rendered_r.value();
        rp.prompt_set_id  = pset_id;
        rp.system_hash    = sha256_prefixed(sys_tmpl.content);
        rp.user_hash      = sha256_prefixed(rp.user_text);
        rp.rendered_hash  = rp.user_hash; // cache key = hash of rendered user prompt

        ExtractionRequestMetadata meta;
        meta.request_id  = detail::make_request_id(req_counter++);
        meta.document_id = chunk.document_id;
        meta.chunk_id    = chunk.id;
        meta.timestamp   = detail::now_iso8601_utc();

        ExtractionResponse resp;
        std::string req_ts = meta.timestamp;

        // Serve from cache if available (no API call, but still parse and write output).
        if (pcfg.cfg.cache_enabled && !pcfg.cfg.replay_mode) {
            auto cached = cache.get(rp.rendered_hash);
            if (cached.is_ok()) {
                resp.raw_response      = cached.value();
                resp.raw_response_hash = sha256_prefixed(cached.value());
                resp.status            = ExtractionStatus::cached;
                resp.provider          = pcfg.provider_cfg.provider;
                resp.model             = pcfg.provider_cfg.model;
                ++result.chunks_skipped;
            }
        }

        if (resp.status != ExtractionStatus::cached) {
            resp    = provider.extract(rp, meta, pcfg.provider_cfg);
            // Persist new response to cache.
            if (pcfg.cfg.cache_enabled && !resp.raw_response.empty() &&
                resp.status == ExtractionStatus::success)
                cache.put(rp.rendered_hash, resp.raw_response);
        }

        std::string resp_ts = detail::now_iso8601_utc();

        // Write call record.
        f_call_recs << detail::call_record_to_json(
            meta, rp, resp, req_ts, resp_ts,
            pcfg.provider_cfg.temperature, pcfg.provider_cfg.max_output_tokens).dump() << "\n";

        if (resp.status != ExtractionStatus::success &&
            resp.status != ExtractionStatus::cached  &&
            resp.status != ExtractionStatus::replayed) {
            result.errors.push_back("Chunk " + chunk.id + ": " + to_string(resp.status) +
                                    " — " + resp.error_message);
            ++result.chunks_processed;
            continue;
        }

        // Optionally record raw response to --record-to directory (fixture generation).
        if (!pcfg.record_to_dir.empty() && !resp.raw_response.empty()) {
            fs::create_directories(pcfg.record_to_dir);
            std::string hex = rp.rendered_hash;
            if (hex.size() > 7 && hex.substr(0, 7) == "sha256:") hex = hex.substr(7);
            std::ofstream rf(pcfg.record_to_dir + "/" + hex + ".json");
            rf << resp.raw_response;
        }

        // Parse and validate the raw response.
        auto parsed = parse_extraction_response(
            resp.raw_response,
            pcfg.allowed_entity_types,
            pcfg.allowed_relation_types,
            meta, resp.provider, resp.model);

        // Log validation errors to rejected_candidates.jsonl.
        for (const auto& ve : parsed.validation_errors) {
            ++result.rejected;
            f_rejected << nlohmann::json({
                {"chunk_id",    chunk.id},
                {"document_id", chunk.document_id},
                {"reason",      ve}
            }).dump() << "\n";
        }

        // Map local_id -> candidate_id for entity cross-reference in relations.
        std::unordered_map<std::string, std::string> local_to_cand;

        for (const auto& e : parsed.response.entities) {
            std::string cand_id = detail::make_candidate_id("cand_ent_", ent_counter++);
            local_to_cand[e.local_id] = cand_id;

            nlohmann::json j = {
                {"candidate_id",      cand_id},
                {"document_id",       chunk.document_id},
                {"chunk_id",          chunk.id},
                {"name",              e.name},
                {"type",              e.type},
                {"description",       e.description},
                {"evidence_text",     e.evidence_text},
                {"confidence",        e.confidence},
                {"provider",          resp.provider},
                {"model",             resp.model},
                {"prompt_set_id",     pset_id},
                {"validation_status", "unvalidated"},
            };
            if (!e.aliases.empty()) j["aliases"] = e.aliases;
            f_entities << j.dump() << "\n";
            ++result.entities_written;
        }

        for (const auto& r : parsed.response.relations) {
            std::string cand_id = detail::make_candidate_id("cand_rel_", rel_counter++);

            nlohmann::json head = nlohmann::json::array();
            for (const auto& id : r.head_local_ids) {
                auto it = local_to_cand.find(id);
                if (it != local_to_cand.end()) head.push_back(it->second);
            }
            nlohmann::json tail = nlohmann::json::array();
            for (const auto& id : r.tail_local_ids) {
                auto it = local_to_cand.find(id);
                if (it != local_to_cand.end()) tail.push_back(it->second);
            }

            f_relations << nlohmann::json({
                {"candidate_id",      cand_id},
                {"document_id",       chunk.document_id},
                {"chunk_id",          chunk.id},
                {"head_candidate_ids", head},
                {"relation_type",     r.relation_type},
                {"tail_candidate_ids", tail},
                {"evidence_text",     r.evidence_text},
                {"confidence",        r.confidence},
                {"provider",          resp.provider},
                {"model",             resp.model},
                {"prompt_set_id",     pset_id},
                {"validation_status", "unvalidated"},
            }).dump() << "\n";
            ++result.relations_written;
        }

        ++result.chunks_processed;
    }

    return result;
}

} // namespace slmkg
