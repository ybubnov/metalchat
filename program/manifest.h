// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <toml.hpp>

#include "digest.h"
#include "http.h"


namespace metalchat {
namespace runtime {


struct architecture {
    static std::string llama3;
};


struct partitioning {
    static std::string consolidated;
    static std::string sharded;
};


struct variant {
    static std::string huggingface;
};


struct model {
    std::string repository;
    std::string variant;
    std::string architecture;
    std::string partitioning;
};


struct manifest {
    static constexpr std::string_view default_name = "manifest.toml";

    using option_key = std::string;
    using option_value = std::string;

    template <typename K, typename V> using optional_map = std::optional<std::map<K, V>>;

    model model;
    optional_map<option_key, option_value> options;

    /// Return a SHA-1 digest of model specification.
    ///
    /// The implementation creates a normalized URL with query parameters as model
    /// specification attributes. And then compute SHA-1 digest from percent-encoded
    /// string representation of the final URL.
    std::string
    id() const
    {
        url u(model.repository);
        u.push_query("variant", model.variant);
        u.push_query("architecture", model.architecture);
        u.push_query("partitioning", model.partitioning);

        return sha1(u);
    }

    std::string
    abbrev_id(std::size_t n = 7) const
    {
        return id().substr(0, n);
    }

    void
    set_option(const option_key& key, const option_value& value)
    {
        using options_type = decltype(options);
        auto o = options.value_or(options_type::value_type());
        o.insert_or_assign(key, value);
        options = o;
    }

    void
    unset_option(const option_key& key)
    {
        if (options) {
            auto& o = options.value();
            if (auto it = o.find(key); it != o.end()) {
                o.erase(it);
            }
            if (o.empty()) {
                options = std::nullopt;
            }
        }
    }

    std::optional<option_value>
    get_option(const option_key& key) const
    {
        return options.and_then([&](auto& c) -> std::optional<option_value> {
            if (auto it = c.find(key); it != c.end()) {
                return it->second;
            }
            return std::nullopt;
        });
    }
};


} // namespace runtime
} // namespace metalchat


TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(
    metalchat::runtime::model, repository, architecture, partitioning, variant
);
TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(metalchat::runtime::manifest, model, options);
