// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <string>

#include <toml.hpp>


namespace metalchat {
namespace program {


struct credential_config {
    std::string username;
    std::string provider;
};


struct config {
    template <typename Key, typename Value>
    using optional_table = std::optional<std::map<Key, Value>>;

    using credential_table = optional_table<std::string, credential_config>;

    /// Credentials contain a list of remotes to download models.
    credential_table credential;

    void
    push_credential(const std::string& url, const credential_config& c)
    {
        auto creds = credential.value_or(credential_table::value_type());
        creds.insert_or_assign(url, c);
        credential = creds;
    }

    void
    pop_credential(const std::string& url)
    {
        if (credential.has_value()) {
            auto creds = credential.value_or(credential_table::value_type());
            if (auto it = creds.find(url); it != creds.end()) {
                creds.erase(it);
            }
            credential = creds;
        }
    }
};

} // namespace program
} // namespace metalchat


TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(metalchat::program::credential_config, username, provider);
TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(metalchat::program::config, credential);


namespace metalchat {
namespace program {


template <typename T> class tomlfile {
public:
    tomlfile(const std::filesystem::path& p)
    : _M_path(p)
    {}

    void
    write(const T& t) const
    {
        auto toml_document = toml::value(t);

        if (toml_document.is_table()) {
            auto& table_ref = toml_document.as_table();

            for (auto& [name, child] : table_ref) {
                if (child.is_table()) {
                    child.as_table_fmt().fmt = toml::table_format::implicit;
                }
            }
        }

        auto toml_bytes = toml::format(toml_document);
        std::string_view toml_bytes_view = toml_bytes;

        while (!toml_bytes_view.empty() && toml_bytes_view.back() == '\n') {
            toml_bytes_view.remove_suffix(1);
        }

        std::ofstream file_stream(_M_path, std::ios::binary | std::ios::trunc);
        file_stream << toml_bytes_view;
    }

    static void
    write(const std::filesystem::path& p, const T& t)
    {
        tomlfile<T> file(p);
        file.write(t);
    }

    T
    read() const
    {
        if (!std::filesystem::exists(_M_path)) {
            std::ofstream file(_M_path);
            file.close();
        }
        auto toml_document = toml::parse(_M_path);
        return toml::get<T>(toml_document);
    }

    static T
    read(const std::filesystem::path& p)
    {
        tomlfile<T> file(p);
        return file.read();
    }

private:
    std::filesystem::path _M_path;
};


} // namespace program
} // namespace metalchat
