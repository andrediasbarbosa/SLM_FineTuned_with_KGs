#pragma once
#include <string>
#include <vector>
#include "core/errors.hpp"

namespace slmkg {

enum class ExtractionStatus {
    success,
    provider_error,
    timeout,
    invalid_json,
    schema_validation_failed,
    empty_response,
    retry_exhausted,
    skipped,
    cached,
    replayed
};

inline std::string to_string(ExtractionStatus s) {
    switch (s) {
        case ExtractionStatus::success:                  return "success";
        case ExtractionStatus::provider_error:           return "provider_error";
        case ExtractionStatus::timeout:                  return "timeout";
        case ExtractionStatus::invalid_json:             return "invalid_json";
        case ExtractionStatus::schema_validation_failed: return "schema_validation_failed";
        case ExtractionStatus::empty_response:           return "empty_response";
        case ExtractionStatus::retry_exhausted:          return "retry_exhausted";
        case ExtractionStatus::skipped:                  return "skipped";
        case ExtractionStatus::cached:                   return "cached";
        case ExtractionStatus::replayed:                 return "replayed";
    }
    return "unknown";
}

struct RawExtractionEntity {
    std::string              local_id;
    std::string              name;
    std::string              type;
    std::vector<std::string> aliases;
    std::string              description;
    std::string              evidence_text;
    double                   confidence = 0.0;
};

struct RawExtractionRelation {
    std::vector<std::string> head_local_ids;
    std::string              relation_type;
    std::vector<std::string> tail_local_ids;
    std::string              evidence_text;
    double                   confidence = 0.0;
};

struct ExtractionResponse {
    ExtractionStatus                   status = ExtractionStatus::success;
    std::vector<RawExtractionEntity>   entities;
    std::vector<RawExtractionRelation> relations;
    std::string raw_response;
    std::string raw_response_hash;
    std::string error_message;
    std::string provider;
    std::string model;
};

// Carries both prompt texts and their hashes for audit / cache keying.
struct RenderedPrompt {
    std::string system_text;
    std::string user_text;
    std::string prompt_set_id;
    std::string system_hash;   // "sha256:..."
    std::string user_hash;     // "sha256:..."
    std::string rendered_hash; // sha256 of user_text — used as cache key
};

struct ExtractionRequestMetadata {
    std::string request_id;
    std::string document_id;
    std::string chunk_id;
    std::string timestamp;
};

struct ProviderConfig {
    std::string provider            = "openai";
    std::string model               = "gpt-4.1-mini";
    double      temperature         = 0.0;
    int         max_output_tokens   = 4096;
    int         max_attempts        = 3;
    int         initial_backoff_ms  = 500;
    int         max_backoff_ms      = 8000;
    int         connect_timeout_ms  = 5000;
    int         request_timeout_ms  = 30000;
};

class IKgExtractionProvider {
public:
    virtual ~IKgExtractionProvider() = default;

    virtual ExtractionResponse extract(
        const RenderedPrompt&            prompt,
        const ExtractionRequestMetadata& metadata,
        const ProviderConfig&            config) = 0;
};

} // namespace slmkg
