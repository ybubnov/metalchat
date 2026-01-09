// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>
#include <format>
#include <iostream>

#include <curl/curl.h>


namespace metalchat {
namespace workspace {


class url {
public:
    using native_handle_type = std::shared_ptr<CURLU>;

    url(const std::string& u);

    std::string
    protocol() const;

    std::string
    host() const;

    std::string
    string() const;

    const native_handle_type
    native_handle() const;

private:
    std::string
    part(CURLUPart p) const;

    native_handle_type _M_url;
};


/// The class represents an abstraction of a remote file located at the specified url.
///
/// The implementation uses libcurl to query the remote file. It's a stateless implementation
/// of the remote object, meaning that if \ref httpfile points to a dynamic file, it's size
/// might change as well.
class httpfile {
public:
    httpfile(const url& u);
    httpfile(const std::string& u);
    ~httpfile();

    template <std::output_iterator<char> OutputIt>
    void
    read(OutputIt output) const
    {
        std::shared_ptr<CURL> curl(curl_easy_init(), curl_easy_cleanup);
        if (curl == nullptr) {
            throw std::runtime_error(
                std::format("httpfile: failed initializing reader for '{}'", _M_url.string())
            );
        }

        auto url = _M_url.string();
        curl_easy_setopt(curl.get(), CURLOPT_URL, url.data());
        curl_easy_setopt(curl.get(), CURLOPT_VERBOSE, 1l);
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 1l);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, read_cb<OutputIt>);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &output);

        auto error = curl_easy_perform(curl.get());
        if (error) {
            throw std::runtime_error(
                std::format("httpfile: failed reading a remote url '{}'", _M_url.string())
            );
        }
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
};


} // namespace workspace
} // namespace metalchat
