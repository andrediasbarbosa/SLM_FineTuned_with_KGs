#pragma once
#include <string>
#include <sstream>
#include <iomanip>
#include <openssl/evp.h>

namespace slmkg {

// Returns lowercase hex SHA256 digest of input (no prefix).
inline std::string sha256_hex(const std::string& input) {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, input.data(), input.size());
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int  len = 0;
    EVP_DigestFinal_ex(ctx, hash, &len);
    EVP_MD_CTX_free(ctx);

    std::ostringstream oss;
    for (unsigned int i = 0; i < len; ++i)
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    return oss.str();
}

// Returns "sha256:<hex>" prefixed digest.
inline std::string sha256_prefixed(const std::string& input) {
    return "sha256:" + sha256_hex(input);
}

} // namespace slmkg
