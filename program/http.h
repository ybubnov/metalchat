// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <iterator>
#include <unordered_map>

#include <curl/curl.h>


namespace metalchat {
namespace runtime {


class url {
public:
    using native_handle_type = CURLU;

    url(const std::string& u);

    std::string
    protocol() const;

    std::string
    host() const;

    std::string
    base() const;

    std::string
    path() const;

    std::string
    string() const;

    const std::shared_ptr<native_handle_type>
    native_handle() const;

    friend url
    operator/(const url& lhs, const std::string& p);

private:
    std::string
    part(CURLUPart p) const;

    std::shared_ptr<native_handle_type> _M_url;
};


/// The class represents an abstraction of a remote file located at the specified url.
///
/// The implementation uses libcurl to query the remote file. It's a stateless implementation
/// of the remote object, meaning that if \ref http_file points to a dynamic file, it's size
/// might change as well.
class http_file {
public:
    http_file(const url& u);
    http_file(const std::string& u);
    ~http_file();

    http_file&
    set_header(const std::string& key, const std::string& value);

    const url&
    location() const;

    template <std::output_iterator<char> OutputIt>
    void
    read(OutputIt output) const
    {
        std::shared_ptr<CURL> curl(curl_easy_init(), curl_easy_cleanup);
        if (curl == nullptr) {
            throw std::runtime_error(
                std::format("http_file: failed initializing reader for '{}'", _M_url.string())
            );
        }

        struct curl_slist* headers_ptr = nullptr;
        for (const auto& [k, v] : _M_headers) {
            auto header = k + ": " + v;
            headers_ptr = curl_slist_append(headers_ptr, header.c_str());
        }

        std::shared_ptr<curl_slist> headers(headers_ptr, curl_slist_free_all);

        auto url = _M_url.string();
        curl_easy_setopt(curl.get(), CURLOPT_URL, url.data());
        curl_easy_setopt(curl.get(), CURLOPT_VERBOSE, 1l);
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 1l);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, read_cb<OutputIt>);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &output);
        curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1l);
        curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());
        curl_easy_setopt(curl.get(), CURLOPT_FAILONERROR, 1l);

        auto error = curl_easy_perform(curl.get());
        if (error == CURLE_OK) {
            return;
        }

        long response_code = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &response_code);

        std::string error_text;
        if (response_code != 0) {
            error_text = std::format("http_file: {}\n", response_code);
        }

        error_text = std::format("{}http_file: {}", error_text, _M_url.string());
        error_text = std::format("{}\nhttp_file: {}", error_text, curl_easy_strerror(error));

        throw std::runtime_error(error_text);
    }

private:
    template <std::output_iterator<char> OutputIt>
    static size_t
    read_cb(void* ptr, size_t count, size_t size, void* stream)
    {
        const auto first = static_cast<char*>(ptr);
        const auto last = first + (count * size) / sizeof(char);

        auto output = reinterpret_cast<OutputIt*>(stream);
        std::copy(first, last, *output);

        return count * size;
    }

    url _M_url;
    std::unordered_map<std::string, std::string> _M_headers;
};


using http_middlware = std::function<void(http_file&)>;


class http_filesystem {
public:
    http_filesystem(const url& base, http_middlware&& middleware);
    http_filesystem(const url& base);

    template <std::output_iterator<char> OutputIt>
    void
    read(const std::string& path, OutputIt output) const
    {
        http_file file(_M_url / path);
        if (_M_middleware) {
            _M_middleware(file);
        }
        file.read(output);
    }

private:
    url _M_url;
    http_middlware _M_middleware;
};


} // namespace runtime
} // namespace metalchat
