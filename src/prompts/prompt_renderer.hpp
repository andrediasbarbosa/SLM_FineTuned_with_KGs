#pragma once
#include <string>
#include <unordered_map>
#include "prompt_template.hpp"
#include "core/errors.hpp"

namespace slmkg {

// Substitutes all {{variable}} occurrences in template content.
// Returns an error if any placeholder present in the template has no
// corresponding entry in vars.
inline Result<std::string> render_prompt(
        const PromptTemplate&                            tmpl,
        const std::unordered_map<std::string, std::string>& vars) {

    // Check for missing variables first.
    for (const auto& name : tmpl.placeholders) {
        if (vars.find(name) == vars.end())
            return Result<std::string>::err("Missing template variable: {{" + name + "}}");
    }

    std::string out = tmpl.content;
    for (const auto& [key, val] : vars) {
        std::string placeholder = "{{" + key + "}}";
        std::string::size_type pos = 0;
        while ((pos = out.find(placeholder, pos)) != std::string::npos) {
            out.replace(pos, placeholder.size(), val);
            pos += val.size();
        }
    }
    return Result<std::string>::ok(std::move(out));
}

} // namespace slmkg
