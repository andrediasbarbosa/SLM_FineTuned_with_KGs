#pragma once
#include <string>
#include "core/errors.hpp"
#include <yaml-cpp/yaml.h>

namespace slmkg {

struct KgExtractionPromptsConfig {
    std::string prompt_set_id      = "kg_extract_promptset_v0.1";
    std::string system_prompt_path = "prompts/kg_system_prompt_v0.1.txt";
    std::string user_prompt_path   = "prompts/kg_user_prompt_v0.1.txt";
};

struct KgExtractionOutputConfig {
    std::string expected_format   = "json";
    std::string schema_path       = "configs/kg_extraction_output_schema.json";
    bool        strict_validation = true;
};

struct KgExtractionConfig {
    std::string provider         = "openai";
    std::string extraction_mode  = "entities_and_relations";
    bool        cache_enabled    = true;
    std::string cache_dir        = ".cache/kg_extraction";
    bool        replay_mode      = false;
    bool        test_mode        = false;   // forces mock provider in CI

    KgExtractionPromptsConfig prompts;
    KgExtractionOutputConfig  output;

    // Provider / retry / timeout (mirrors ProviderConfig fields for convenience)
    std::string model                  = "gpt-4.1-mini";
    double      temperature            = 0.0;
    int         max_output_tokens      = 4096;
    int         max_attempts           = 3;
    int         initial_backoff_ms     = 500;
    int         max_backoff_ms         = 8000;
    int         connect_timeout_ms     = 5000;
    int         request_timeout_ms     = 30000;
    int         max_concurrent_requests = 4;
};

inline Result<KgExtractionConfig> load_kg_extraction_config(const std::string& path) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        YAML::Node c = root["kg_extraction"];
        if (!c) return Result<KgExtractionConfig>::err("Missing 'kg_extraction' key in " + path);

        KgExtractionConfig cfg;
        if (c["provider"])        cfg.provider        = c["provider"].as<std::string>();
        if (c["model"])           cfg.model           = c["model"].as<std::string>();
        if (c["extraction_mode"]) cfg.extraction_mode = c["extraction_mode"].as<std::string>();
        if (c["cache_enabled"])   cfg.cache_enabled    = c["cache_enabled"].as<bool>();
        if (c["cache_dir"])       cfg.cache_dir        = c["cache_dir"].as<std::string>();
        if (c["replay_mode"])     cfg.replay_mode      = c["replay_mode"].as<bool>();
        if (c["test_mode"])       cfg.test_mode        = c["test_mode"].as<bool>();

        if (YAML::Node p = c["prompts"]) {
            if (p["prompt_set_id"])      cfg.prompts.prompt_set_id      = p["prompt_set_id"].as<std::string>();
            if (p["system_prompt_path"]) cfg.prompts.system_prompt_path = p["system_prompt_path"].as<std::string>();
            if (p["user_prompt_path"])   cfg.prompts.user_prompt_path   = p["user_prompt_path"].as<std::string>();
        }

        if (YAML::Node o = c["output"]) {
            if (o["expected_format"])   cfg.output.expected_format   = o["expected_format"].as<std::string>();
            if (o["schema_path"])       cfg.output.schema_path       = o["schema_path"].as<std::string>();
            if (o["strict_validation"]) cfg.output.strict_validation = o["strict_validation"].as<bool>();
        }

        if (YAML::Node r = c["retry"]) {
            if (r["max_attempts"])       cfg.max_attempts       = r["max_attempts"].as<int>();
            if (r["initial_backoff_ms"]) cfg.initial_backoff_ms = r["initial_backoff_ms"].as<int>();
            if (r["max_backoff_ms"])     cfg.max_backoff_ms     = r["max_backoff_ms"].as<int>();
        }

        if (YAML::Node t = c["timeout"]) {
            if (t["connect_ms"]) cfg.connect_timeout_ms = t["connect_ms"].as<int>();
            if (t["request_ms"]) cfg.request_timeout_ms = t["request_ms"].as<int>();
        }

        if (YAML::Node cc = c["concurrency"])
            if (cc["max_concurrent_requests"])
                cfg.max_concurrent_requests = cc["max_concurrent_requests"].as<int>();

        return Result<KgExtractionConfig>::ok(std::move(cfg));
    } catch (const YAML::Exception& e) {
        return Result<KgExtractionConfig>::err(
            std::string("YAML error loading ") + path + ": " + e.what());
    }
}

} // namespace slmkg
