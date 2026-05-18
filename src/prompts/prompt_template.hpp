#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <regex>
#include "core/errors.hpp"

namespace slmkg {

struct PromptTemplate {
    std::string              path;
    std::string              content;
    std::vector<std::string> placeholders;  // deduplicated {{variable}} names
};

// Loads a prompt template file and extracts all {{variable}} placeholder names.
inline Result<PromptTemplate> load_prompt_template(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        return Result<PromptTemplate>::err("Cannot open prompt file: " + path);

    std::ostringstream ss;
    ss << f.rdbuf();

    PromptTemplate tmpl;
    tmpl.path    = path;
    tmpl.content = ss.str();

    // Extract placeholder names (deduplicated, insertion-ordered).
    std::regex re(R"(\{\{([A-Za-z_][A-Za-z0-9_]*)\}\})");
    std::sregex_iterator it(tmpl.content.begin(), tmpl.content.end(), re);
    std::sregex_iterator end;
    std::vector<std::string> seen;
    for (; it != end; ++it) {
        std::string name = (*it)[1].str();
        if (std::find(seen.begin(), seen.end(), name) == seen.end()) {
            seen.push_back(name);
            tmpl.placeholders.push_back(name);
        }
    }

    return Result<PromptTemplate>::ok(std::move(tmpl));
}

} // namespace slmkg
