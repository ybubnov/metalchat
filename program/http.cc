// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <format>

#include "http.h"


namespace metalchat {
namespace program {


url::url(const std::string& u)
: _M_url(curl_url())
{
    auto error = curl_url_set(_M_url, CURLUPART_URL, u.data(), 0);
    if (error) {
        throw std::invalid_argument(std::format("url: failed to parse url '{}'", u));
    }
}


url::~url()
{
    if (_M_url) {
        curl_url_cleanup(_M_url);
    }
    _M_url = nullptr;
}


std::string
url::part(CURLUPart p) const
{
    char* part = nullptr;
    auto error = curl_url_get(_M_url, p, &part, 0);
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


} // namespace program
} // namespace metalchat
