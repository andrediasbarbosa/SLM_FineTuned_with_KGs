#pragma once
#include <string>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include "provider_interface.hpp"
#include "core/hashing.hpp"

namespace slmkg {
namespace fs = std::filesystem;

class MockKgExtractionProvider : public IKgExtractionProvider {
public:
    explicit MockKgExtractionProvider(const std::string& fixture_dir)
        : fixture_dir_(fixture_dir)
    {
        load_fixtures();
    }

    ExtractionResponse extract(
            const RenderedPrompt&            prompt,
            const ExtractionRequestMetadata& /*metadata*/,
            const ProviderConfig&            /*config*/) override
    {
        std::string hash_hex = strip_prefix(prompt.rendered_hash);

        // Error-path fixture: __error__{sha256}.json
        auto err_it = fixtures_.find("__error__" + hash_hex);
        if (err_it != fixtures_.end()) {
            ExtractionResponse r;
            r.provider          = "mock";
            r.model             = "mock";
            r.raw_response      = err_it->second;
            r.raw_response_hash = sha256_prefixed(r.raw_response);
            r.status            = ExtractionStatus::provider_error;
            r.error_message     = "mock injected error";
            return r;
        }

        auto it = fixtures_.find(hash_hex);
        if (it != fixtures_.end()) {
            ExtractionResponse r;
            r.provider          = "mock";
            r.model             = "mock";
            r.raw_response      = it->second;
            r.raw_response_hash = sha256_prefixed(r.raw_response);
            r.status            = ExtractionStatus::cached;
            return r;
        }

        // No fixture for this hash — return empty and warn (do not error).
        std::cerr << "[WARN] MockProvider: no fixture for hash " << hash_hex
                  << "; returning empty result\n";
        ExtractionResponse r;
        r.provider          = "mock";
        r.model             = "mock";
        r.raw_response      = R"({"entities":[],"relations":[]})";
        r.raw_response_hash = sha256_prefixed(r.raw_response);
        r.status            = ExtractionStatus::success;
        return r;
    }

private:
    std::string fixture_dir_;
    std::unordered_map<std::string, std::string> fixtures_; // stem -> content

    static std::string strip_prefix(const std::string& hash) {
        if (hash.size() > 7 && hash.substr(0, 7) == "sha256:") return hash.substr(7);
        return hash;
    }

    void load_fixtures() {
        if (!fs::exists(fixture_dir_)) return;
        for (const auto& entry : fs::directory_iterator(fixture_dir_)) {
            if (entry.path().extension() != ".json") continue;
            std::ifstream f(entry.path());
            std::ostringstream ss;
            ss << f.rdbuf();
            fixtures_[entry.path().stem().string()] = ss.str();
        }
    }
};

} // namespace slmkg
