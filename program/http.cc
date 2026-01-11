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


const std::shared_ptr<url::native_handle_type>
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
url::base() const
{
    return std::format("{}://{}", protocol(), host());
}


std::string
url::path() const
{
    return part(CURLUPART_PATH);
}


std::string
url::string() const
{
    return part(CURLUPART_URL);
}


url
operator/(const url& lhs, const std::string& p)
{
    return url(lhs.string() + "/" + p);
}


http_file::http_file(const url& u)
: _M_url(u),
  _M_headers()
{
    auto error = curl_global_init(CURL_GLOBAL_ALL);
    if (error) {
        throw std::runtime_error("http_file: unable to initialize file");
    }
}


http_file::http_file(const std::string& u)
: http_file(url(u))
{}


http_file&
http_file::set_header(const std::string& key, const std::string& value)
{
    _M_headers.insert_or_assign(key, value);
    return *this;
}


http_file::~http_file() { curl_global_cleanup(); }


http_filesystem::http_filesystem(const url& base, http_middlware&& middleware)
: _M_url(base),
  _M_middleware(std::move(middleware))
{}


http_filesystem::http_filesystem(const url& base)
: _M_url(base),
  _M_middleware(nullptr)
{}


} // namespace runtime
} // namespace metalchat
