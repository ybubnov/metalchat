// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <chrono>
#include <thread>

#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <unordered_map>

#include <curl/curl.h>
#include <metalchat/container.h>


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

    std::size_t
    size() const
    {
        auto handle_ptr = make_handle();
        curl_easy_setopt(handle_ptr.get(), CURLOPT_NOBODY, 1l);

        handle_ptr = round_trip(handle_ptr);

        curl_off_t content_length;
        auto error = curl_easy_getinfo(
            handle_ptr.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length
        );

        if (error) {
            throw std::runtime_error("http_file: unknown file size");
        }

        return content_length;
    }

    template <std::output_iterator<char> OutputIt>
    void
    read(OutputIt& output) const
    {
        auto handle_ptr = make_handle();

        curl_easy_setopt(handle_ptr.get(), CURLOPT_WRITEFUNCTION, write_callback<OutputIt>);
        curl_easy_setopt(handle_ptr.get(), CURLOPT_WRITEDATA, &output);

        round_trip(handle_ptr);
    }

private:
    using handle_type = CURL;
    using handle_pointer = std::shared_ptr<CURL>;

    handle_pointer
    make_handle() const
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

    handle_pointer
    use_headers(handle_pointer handle_ptr) const
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

    handle_pointer
    round_trip(handle_pointer handle_ptr) const
    {
        auto error = curl_easy_perform(handle_ptr.get());
        return throw_on_error(handle_ptr, error);
    }

    handle_pointer
    throw_on_error(handle_pointer handle_ptr, CURLcode error_code) const
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

    template <std::output_iterator<char> OutputIt>
    static size_t
    write_callback(void* ptr, size_t count, size_t size, void* data)
    {
        const auto first = static_cast<char*>(ptr);
        const auto last = first + (count * size) / sizeof(char);

        auto output = reinterpret_cast<OutputIt*>(data);
        std::copy(first, last, *output);

        return count * size;
    }

    url _M_url;
    std::unordered_map<std::string, std::string> _M_headers;
};


using http_middleware = std::function<void(http_file&)>;


/// The HTTP middleware that performs bearer authentication of HTTP requests.
///
/// One the base url of the \ref http_file is present in the secret provider, it
/// will be used to form an 'Authorization' header. If secret is not present, the
/// request remains unchanged.
template <typename SecretProvider> class http_bearer_auth {
public:
    http_bearer_auth(SecretProvider&& secrets = SecretProvider())
        requires std::is_default_constructible_v<SecretProvider>
    : _M_secrets(secrets)
    {}

    http_bearer_auth(const SecretProvider& secrets)
    : _M_secrets(secrets)
    {}

    void
    operator()(http_file& file)
    {
        auto base_url = file.location().base();
        if (auto secret = _M_secrets.load(base_url); secret) {
            file.set_header("Authorization", std::format("Bearer {}", secret.value()));
        }
    }

private:
    SecretProvider _M_secrets;
};


class http_filesystem {
public:
    http_filesystem(const url& base);
    http_filesystem(const url& base, const http_middleware& middleware);

    void
    add_middleware(http_middleware&& middleware)
    {
        _M_middleware.emplace_back(std::move(middleware));
    }

    void
    use_middleware(http_file& file) const
    {
        for (auto& middleware : _M_middleware) {
            middleware(file);
        }
    }

    void
    read(const std::string& path, std::ostream& output) const
    {
        http_file file(_M_url / path);
        use_middleware(file);
        read(file, std::ostream_iterator<char>(output));
    }

protected:
    virtual void
    read(http_file& file, std::ostream_iterator<char> output) const
    {
        file.read(output);
    }

    url _M_url;
    std::vector<http_middleware> _M_middleware;
};


template <typename T, std::output_iterator<T> Iterator> class tracking_iterator {
public:
    using iterator_category = Iterator::iterator_category;
    using value_type = Iterator::value_type;
    using difference_type = Iterator::difference_type;
    using pointer = Iterator::pointer;
    using reference = Iterator::reference;
    using char_type = Iterator::char_type;
    using traits_type = Iterator::traits_type;

    tracking_iterator(
        const Iterator& iterator,
        const std::string& name,
        std::size_t size,
        std::size_t precision = 0
    )
    : _M_iter(iterator),
      _M_name(name),
      _M_cur(std::make_shared<std::size_t>(0)),
      _M_pre(std::make_shared<std::size_t>(0)),
      _M_size(size),
      _M_precision(100 * std::pow(10.0, precision))
    {}

    tracking_iterator(const tracking_iterator&) = default;

    tracking_iterator&
    operator=(const T& value)
    {
        _M_iter = value;
        return *this;
    }

    tracking_iterator&
    operator*()
    {
        return *this;
    }

    tracking_iterator&
    operator++()
    {
        advance();
        return *this;
    }

    tracking_iterator&
    operator++(int)
    {
        advance();
        return *this;
    }

private:
    static constexpr std::array<std::string_view, 5> memory_units = {
        "B", "KiB", "MiB", "GiB", "TiB"
    };

    std::string
    format_bytes(std::size_t size) const
    {
        std::size_t i = 0;
        double bytes = static_cast<double>(size);
        while (bytes >= 1024 && i < memory_units.size() - 1) {
            bytes /= 1024;
            i++;
        }

        return std::format("{:.2f} {}", bytes, std::string(memory_units[i]));
    }

    void
    render_indicator(std::size_t next)
    {
        *_M_pre = next;
        std::cout << "\33[2K\rDownloading " << _M_name << ' ';
        std::cout << format_bytes(*_M_cur) << '/' << format_bytes(_M_size);
        std::cout << std::flush;
    }

    void
    advance()
    {
        _M_iter++;
        (*_M_cur)++;

        const auto next = std::size_t(_M_precision * double(*_M_cur) / double(_M_size));

        if ((*_M_pre) - next > 0) {
            render_indicator(next);
        }
        if (*_M_cur == _M_size) {
            render_indicator(next);
            std::cout << std::endl;
        }
    }

    Iterator _M_iter;
    std::string _M_name;
    std::shared_ptr<std::size_t> _M_cur;
    std::shared_ptr<std::size_t> _M_pre;
    std::size_t _M_size;
    std::size_t _M_precision;
};


struct http_tracking_filesystem : public http_filesystem {
    using http_filesystem::add_middleware;
    using http_filesystem::http_filesystem;
    using http_filesystem::read;
    using http_filesystem::use_middleware;

protected:
    using output_iterator = std::ostream_iterator<char>;

    void
    read(http_file& file, output_iterator output) const override
    {
        using iterator = tracking_iterator<char, output_iterator>;
        auto file_size = file.size();
        auto file_path = std::filesystem::path(file.location().path());

        auto tracking_output = iterator(output, file_path.filename(), file_size);
        file.read(tracking_output);
    }
};


} // namespace runtime
} // namespace metalchat
