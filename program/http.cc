// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <format>

#include "http.h"


namespace metalchat {
namespace runtime {


url::url(const std::string& u)
: _M_url(curl_url(), curl_url_cleanup)
{
    auto error = curl_url_set(_M_url.get(), CURLUPART_URL, u.data(), 0);
    if (error) {
        throw std::invalid_argument(std::format("url: failed to parse url '{}'", u));
    }
}


const url::native_handle_type
url::native_handle() const
{
    return _M_url;
}


std::string
url::part(CURLUPart p) const
{
    char* part = nullptr;
    auto error = curl_url_get(_M_url.get(), p, &part, 0);
    if (error || !part) {
        throw std::invalid_argument("url: failed to extract part of url");
    }

    std::string part_str(part);
    curl_free(part);

    return part_str;
}


std::string
url::protocol() const
{
    return part(CURLUPART_SCHEME);
}


std::string
url::host() const
{
    return part(CURLUPART_HOST);
}


std::string
url::string() const
{
    return part(CURLUPART_URL);
}


httpfile::httpfile(const url& u)
: _M_url(u)
{
    auto error = curl_global_init(CURL_GLOBAL_ALL);
    if (error) {
        throw std::runtime_error("httpfile: unable to initialize file");
    }
}

httpfile::httpfile(const std::string& u)
: httpfile(url(u))
{}


httpfile::~httpfile() { curl_global_cleanup(); }


} // namespace runtime
} // namespace metalchat
