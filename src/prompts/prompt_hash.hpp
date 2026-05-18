#pragma once
#include <string>
#include "core/hashing.hpp"

namespace slmkg {

// Returns "sha256:<hex>" hash of the rendered prompt string.
inline std::string hash_prompt(const std::string& rendered) {
    return sha256_prefixed(rendered);
}

} // namespace slmkg
