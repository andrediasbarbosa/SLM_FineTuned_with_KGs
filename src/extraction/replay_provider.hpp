#pragma once
#include "provider_interface.hpp"
#include "extraction_cache.hpp"
#include "core/hashing.hpp"

namespace slmkg {

class ReplayKgExtractionProvider : public IKgExtractionProvider {
public:
    explicit ReplayKgExtractionProvider(const std::string& cache_dir)
        : cache_(cache_dir) {}

    ExtractionResponse extract(
            const RenderedPrompt&            prompt,
            const ExtractionRequestMetadata& /*metadata*/,
            const ProviderConfig&            /*config*/) override
    {
        auto r = cache_.get(prompt.rendered_hash);
        if (r.is_err()) {
            ExtractionResponse resp;
            resp.provider      = "replay";
            resp.model         = "replay";
            resp.status        = ExtractionStatus::provider_error;
            resp.error_message = "Replay cache miss: " + prompt.rendered_hash;
            return resp;
        }

        ExtractionResponse resp;
        resp.provider          = "replay";
        resp.model             = "replay";
        resp.raw_response      = r.value();
        resp.raw_response_hash = sha256_prefixed(r.value());
        resp.status            = ExtractionStatus::replayed;
        return resp;
    }

private:
    ExtractionCache cache_;
};

} // namespace slmkg
