#pragma once
#include <string>
#include <unordered_map>
#include "prompt_hash.hpp"
#include "core/hashing.hpp"
#include <nlohmann/json.hpp>

namespace slmkg {

struct PromptManifestRecord {
    std::string prompt_set_id;
    std::string system_prompt_path;
    std::string user_prompt_path;
    std::string system_prompt_hash;
    std::string user_prompt_hash;
    std::string rendered_prompt_hash;
    std::unordered_map<std::string, std::string> template_variables;
};

// Builds a PromptManifestRecord from the constituent parts.
// system_content and user_content are the raw file contents (not rendered).
// rendered is the fully rendered user prompt.
inline PromptManifestRecord make_prompt_manifest(
        const std::string& prompt_set_id,
        const std::string& system_prompt_path,
        const std::string& system_content,
        const std::string& user_prompt_path,
        const std::string& user_content,
        const std::string& rendered,
        const std::unordered_map<std::string, std::string>& template_variables) {
    PromptManifestRecord r;
    r.prompt_set_id        = prompt_set_id;
    r.system_prompt_path   = system_prompt_path;
    r.user_prompt_path     = user_prompt_path;
    r.system_prompt_hash   = sha256_prefixed(system_content);
    r.user_prompt_hash     = sha256_prefixed(user_content);
    r.rendered_prompt_hash = hash_prompt(rendered);
    r.template_variables   = template_variables;
    return r;
}

inline nlohmann::json to_json(const PromptManifestRecord& r) {
    nlohmann::json vars = nlohmann::json::object();
    for (const auto& [k, v] : r.template_variables)
        vars[k] = v;

    return {
        {"prompt_set_id",        r.prompt_set_id},
        {"system_prompt_path",   r.system_prompt_path},
        {"user_prompt_path",     r.user_prompt_path},
        {"system_prompt_hash",   r.system_prompt_hash},
        {"user_prompt_hash",     r.user_prompt_hash},
        {"rendered_prompt_hash", r.rendered_prompt_hash},
        {"template_variables",   vars},
    };
}

} // namespace slmkg
