#pragma once
#include <string>
#include <thread>
#include <chrono>
#include <cassert>
#include <cstdlib>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include "provider_interface.hpp"
#include "core/hashing.hpp"
#include "http_utils.hpp"

namespace slmkg {

class GeminiKgExtractionProvider : public IKgExtractionProvider {
public:
    GeminiKgExtractionProvider() {
        const char* tm = std::getenv("SLMKG_TEST_MODE");
        assert((!tm || std::string(tm) != "1") &&
               "GeminiKgExtractionProvider must not be used when SLMKG_TEST_MODE=1");
    }

    ExtractionResponse extract(
            const RenderedPrompt&            prompt,
            const ExtractionRequestMetadata& /*metadata*/,
            const ProviderConfig&            config) override
    {
        const char* key_env = std::getenv("GEMINI_API_KEY");
        if (!key_env || std::string(key_env).empty()) {
            ExtractionResponse r;
            r.provider = "gemini"; r.model = config.model;
            r.status        = ExtractionStatus::provider_error;
            r.error_message = "GEMINI_API_KEY not set";
            return r;
        }

        std::string model = config.model.empty() ? "gemini-2.0-flash" : config.model;
        std::string url   = "https://generativelanguage.googleapis.com/v1beta/models/"
                          + model + ":generateContent?key=" + std::string(key_env);

        nlohmann::json body = {
            {"system_instruction", {{"parts", {{{"text", prompt.system_text}}}}}},
            {"contents", nlohmann::json::array({
                {{"role","user"}, {"parts", {{{"text", prompt.user_text}}}}}
            })},
            {"generationConfig", {
                {"temperature",     config.temperature},
                {"maxOutputTokens", config.max_output_tokens}
            }}
        };

        return with_backoff(body.dump(), url, config, model);
    }

private:
    ExtractionResponse with_backoff(const std::string& body,
                                    const std::string& url,
                                    const ProviderConfig& cfg,
                                    const std::string& model)
    {
        int backoff_ms = cfg.initial_backoff_ms;
        for (int attempt = 0; attempt < cfg.max_attempts; ++attempt) {
            if (attempt > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            auto r = do_request(body, url, cfg, model);
            if (r.status == ExtractionStatus::success       ||
                r.status == ExtractionStatus::invalid_json  ||
                r.status == ExtractionStatus::schema_validation_failed)
                return r;
            backoff_ms = std::min(backoff_ms * 2, cfg.max_backoff_ms);
        }
        ExtractionResponse r;
        r.provider = "gemini"; r.model = model;
        r.status        = ExtractionStatus::retry_exhausted;
        r.error_message = "All retry attempts exhausted";
        return r;
    }

    ExtractionResponse do_request(const std::string& body,
                                   const std::string& url,
                                   const ProviderConfig& cfg,
                                   const std::string& model)
    {
        ExtractionResponse resp;
        resp.provider = "gemini"; resp.model = model;

        CURL* curl = curl_easy_init();
        if (!curl) {
            resp.status = ExtractionStatus::provider_error;
            resp.error_message = "curl_easy_init failed";
            return resp;
        }

        std::string buf;
        struct curl_slist* hdrs = nullptr;
        hdrs = curl_slist_append(hdrs, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL,           url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER,    hdrs);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS,    body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, detail::curl_write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA,     &buf);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(cfg.connect_timeout_ms));
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS,    static_cast<long>(cfg.request_timeout_ms));

        CURLcode res = curl_easy_perform(curl);
        curl_slist_free_all(hdrs);
        curl_easy_cleanup(curl);

        if (res == CURLE_OPERATION_TIMEDOUT) {
            resp.status = ExtractionStatus::timeout;
            resp.error_message = "Request timed out";
            return resp;
        }
        if (res != CURLE_OK) {
            resp.status = ExtractionStatus::provider_error;
            resp.error_message = std::string("curl: ") + curl_easy_strerror(res);
            return resp;
        }

        auto j = nlohmann::json::parse(buf, nullptr, false);
        if (j.is_discarded()) {
            resp.status = ExtractionStatus::invalid_json;
            resp.error_message = "Invalid JSON in Gemini response";
            resp.raw_response  = buf;
            return resp;
        }
        if (j.contains("error")) {
            resp.status = ExtractionStatus::provider_error;
            resp.error_message = j["error"].value("message", "Gemini error");
            return resp;
        }
        try {
            std::string content =
                j["candidates"][0]["content"]["parts"][0]["text"].get<std::string>();
            resp.raw_response      = content;
            resp.raw_response_hash = sha256_prefixed(content);
            resp.status            = ExtractionStatus::success;
        } catch (...) {
            resp.status = ExtractionStatus::invalid_json;
            resp.error_message = "Unexpected Gemini response structure";
        }
        return resp;
    }
};

} // namespace slmkg
