#pragma once
#include <cstddef>
#include <string>

namespace slmkg::detail {

inline std::size_t curl_write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
    auto* buf = static_cast<std::string*>(userdata);
    buf->append(ptr, size * nmemb);
    return size * nmemb;
}

} // namespace slmkg::detail
