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


url&
url::push_query(const std::string& key, const std::string& value)
{
    auto flags = CURLU_APPENDQUERY | CURLU_URLENCODE;
    auto query = std::format("{}={}", key, value);
    auto error = curl_url_set(_M_url.get(), CURLUPART_QUERY, query.data(), flags);
    if (error) {
        throw std::invalid_argument(std::format("url: failed to push query parameter '{}'", query));
    }
    return *this;
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


http_file::http_file(const class url& u)
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


const url&
http_file::location() const
{
    return _M_url;
}


std::size_t
http_file::size() const
{
    auto handle_ptr = make_handle();
    curl_easy_setopt(handle_ptr.get(), CURLOPT_NOBODY, 1l);

    handle_ptr = round_trip(handle_ptr);

    curl_off_t content_length;
    auto error =
        curl_easy_getinfo(handle_ptr.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);

    if (error) {
        throw std::runtime_error("http_file: unknown file size");
    }

    return content_length;
}


bool
http_file::exists() const
{
    auto handle_ptr = make_handle();

    curl_easy_setopt(handle_ptr.get(), CURLOPT_NOBODY, 1l);
    return curl_easy_perform(handle_ptr.get()) == CURLE_OK;
}


http_file::handle_pointer
http_file::make_handle() const
{
    handle_pointer handle_ptr(curl_easy_init(), curl_easy_cleanup);
    if (handle_ptr == nullptr) {
        throw std::runtime_error(
            std::format("http_file: failed initializing reader for '{}'", _M_url.string())
        );
    }

    auto url = _M_url.string();
    curl_easy_setopt(handle_ptr.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(handle_ptr.get(), CURLOPT_VERBOSE, 0l);
    curl_easy_setopt(handle_ptr.get(), CURLOPT_NOPROGRESS, 1l);
    curl_easy_setopt(handle_ptr.get(), CURLOPT_FOLLOWLOCATION, 1l);
    curl_easy_setopt(handle_ptr.get(), CURLOPT_FAILONERROR, 1l);

    handle_ptr = use_headers(handle_ptr);
    return handle_ptr;
}


http_file::handle_pointer
http_file::use_headers(handle_pointer handle_ptr) const
{
    struct curl_slist* list_ptr = nullptr;
    for (const auto& [k, v] : _M_headers) {
        auto header = k + ": " + v;
        list_ptr = curl_slist_append(list_ptr, header.c_str());
    }

    std::shared_ptr<curl_slist> headers_ptr(list_ptr, curl_slist_free_all);
    curl_easy_setopt(handle_ptr.get(), CURLOPT_HTTPHEADER, headers_ptr.get());

    return metalchat::make_pointer_alias(handle_ptr, headers_ptr);
}


http_file::handle_pointer
http_file::round_trip(handle_pointer handle_ptr) const
{
    auto error = curl_easy_perform(handle_ptr.get());
    return throw_on_error(handle_ptr, error);
}


http_file::handle_pointer
http_file::throw_on_error(handle_pointer handle_ptr, CURLcode error_code) const
{
    if (error_code == CURLE_OK) {
        return handle_ptr;
    }

    long response_code = 0;
    curl_easy_getinfo(handle_ptr.get(), CURLINFO_RESPONSE_CODE, &response_code);

    std::string error_text;
    if (response_code != 0) {
        error_text = std::format("http_file: {}\n", response_code);
    }

    error_text = std::format("{}http_file: {}", error_text, _M_url.string());
    error_text = std::format("{}\nhttp_file: {}", error_text, curl_easy_strerror(error_code));

    throw std::runtime_error(error_text);
}


http_file::~http_file() { curl_global_cleanup(); }


http_filesystem::http_filesystem(const url& base)
: _M_url(base),
  _M_middleware()
{}


http_filesystem::http_filesystem(const url& base, const http_middleware& middleware)
: _M_url(base),
  _M_middleware({middleware})
{}


} // namespace runtime
} // namespace metalchat
