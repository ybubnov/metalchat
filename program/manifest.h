// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <map>
#include <string_view>
#include <variant>

#include <toml.hpp>


using primitive_variant = std::variant<bool, int, float, std::string>;


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


struct model_section {
    std::string repository;
    std::string variant;
    std::string architecture;
    std::string partitioning;
};


struct prompt_section {
    /// A file path that defines a prompt for a system message.
    std::string system;
};


/// Environment section defines strategy and parameters of running a model.
struct environment_section {
    using option_key = std::string;
    using option_value = primitive_variant;
    using options_section = std::map<option_key, option_value>;

    std::optional<std::size_t> max_sequence_length;
    std::optional<std::string> placement;
    std::optional<std::vector<options_section>> sampling;
};


struct manifest {
    static constexpr std::string_view default_name = "manifest.toml";
    static constexpr std::string_view workspace_name = "metalchat.toml";

    using option_key = std::string;
    using option_value = primitive_variant;
    using options_section = std::map<option_key, option_value>;

    model_section model;
    std::optional<options_section> options;
    std::optional<prompt_section> prompt;
    std::optional<environment_section> environment;

    /// Return a SHA-1 digest of model specification.
    ///
    /// The implementation creates a normalized URL with query parameters as model
    /// specification attributes. And then compute SHA-1 digest from percent-encoded
    /// string representation of the final URL.
    std::string
    id() const;

    /// Return an abbreviated version of the manifest identifier.
    ///
    /// \param n specifies the length of the resulting abbreviated identifier.
    std::string
    abbrev_id(std::size_t n = 7) const;

    /// Retrieve the system prompt from the configured file relative to the `root`.
    std::optional<std::string>
    system_prompt(const std::filesystem::path& root) const;

    /// Set the model option value. The list of supported model options depends on
    /// the specific architecture and implementation. This function does not validate
    /// support of the set option.
    void
    set_option(const option_key& key, const option_value& value);

    /// Remove a specified option from the manifest, method does not throw an exception
    /// when the key is missing.
    void
    unset_option(const option_key& key);

    /// Retrieve an option from the manifest, if present.
    std::optional<option_value>
    get_option(const option_key& key) const;
};


} // namespace runtime
} // namespace metalchat


template <> struct toml::into<primitive_variant> {
    template <typename TypeConfig>
    static toml::basic_value<TypeConfig>
    into_toml(const primitive_variant& v)
    {
        return std::visit([](auto&& value) { return toml::value(value); }, v);
    }
};

template <> struct toml::from<primitive_variant> {
    static primitive_variant
    from_toml(const toml::value& v)
    {
        if (v.is_boolean()) {
            return v.as_boolean();
        }
        if (v.is_integer()) {
            return static_cast<int>(v.as_integer());
        }
        if (v.is_floating()) {
            return static_cast<float>(v.as_floating());
        }
        if (v.is_string()) {
            return v.as_string();
        }

        throw toml::type_error("failed paring a value", v.location());
    }
};


TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(
    metalchat::runtime::model_section, repository, architecture, partitioning, variant
);
TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(
    metalchat::runtime::environment_section, max_sequence_length, placement, sampling
);
TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(metalchat::runtime::prompt_section, system);
TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(metalchat::runtime::manifest, model, options, prompt);
