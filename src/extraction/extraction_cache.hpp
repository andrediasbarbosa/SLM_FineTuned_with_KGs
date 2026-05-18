#pragma once
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "core/errors.hpp"

namespace slmkg {
namespace fs = std::filesystem;

class ExtractionCache {
public:
    explicit ExtractionCache(const std::string& cache_dir) : cache_dir_(cache_dir) {}

    bool has(const std::string& rendered_hash) const {
        return fs::exists(cache_path(rendered_hash));
    }

    Result<std::string> get(const std::string& rendered_hash) const {
        std::string p = cache_path(rendered_hash);
        if (!fs::exists(p))
            return Result<std::string>::err("Cache miss: " + rendered_hash);
        std::ifstream f(p);
        std::ostringstream ss;
        ss << f.rdbuf();
        return Result<std::string>::ok(ss.str());
    }

    Result<void> put(const std::string& rendered_hash, const std::string& raw_response) {
        try {
            fs::create_directories(cache_dir_);
            std::ofstream f(cache_path(rendered_hash));
            if (!f.is_open())
                return Result<void>::err("Cannot write cache entry for " + rendered_hash);
            f << raw_response;
            return Result<void>::ok();
        } catch (const std::exception& e) {
            return Result<void>::err(std::string("Cache write error: ") + e.what());
        }
    }

private:
    std::string cache_dir_;

    std::string cache_path(const std::string& rendered_hash) const {
        std::string hex = rendered_hash;
        if (hex.size() > 7 && hex.substr(0, 7) == "sha256:") hex = hex.substr(7);
        return cache_dir_ + "/" + hex + ".json";
    }
};

} // namespace slmkg
