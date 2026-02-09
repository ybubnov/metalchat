// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "manifest.h"
#include "digest.h"
#include "http.h"


namespace metalchat {
namespace runtime {

std::string architecture::llama3 = "llama3";
std::string variant::huggingface = "huggingface";
std::string partitioning::sharded = "sharded";
std::string partitioning::consolidated = "consolidated";


std::string
manifest::id() const
{
    url u(model.repository);
    u.push_query("variant", model.variant);
    u.push_query("architecture", model.architecture);
    u.push_query("partitioning", model.partitioning);

    return sha1(u);
}


std::string
manifest::abbrev_id(std::size_t n) const
{
    return id().substr(0, n);
}


std::optional<std::string>
manifest::system_prompt(const std::filesystem::path& scope_path) const
{
    if (!prompt) {
        return std::nullopt;
    }

    auto& prompt_value = prompt.value();
    if (prompt_value.system.empty()) {
        return std::nullopt;
    }
    auto prompt_path = std::filesystem::path(prompt_value.system);
    if (prompt_path.is_relative()) {
        prompt_path = scope_path / prompt_path;
    }
    if (!std::filesystem::exists(prompt_path)) {
        throw std::runtime_error(
            std::format("manifest: system prompt '{}' not found", prompt_path.string())
        );
    }
    std::ifstream prompt_file(prompt_path, std::ios::binary | std::ios::in);
    if (!prompt_file.is_open()) {
        throw std::runtime_error(
            std::format("manifest: failed opening a prompt file '{}'", prompt_path.string())
        );
    }
    std::stringstream prompt_stream;

    prompt_stream << prompt_file.rdbuf();
    return prompt_stream.str();
}


void
manifest::set_option(const option_key& key, const option_value& value)
{
    using options_type = decltype(options);
    auto o = options.value_or(options_type::value_type());
    o.insert_or_assign(key, value);
    options = o;
}


void
manifest::unset_option(const option_key& key)
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
manifest::get_option(const option_key& key) const
{
    if (options) {
        auto& o = options.value();
        if (auto it = o.find(key); it != o.end()) {
            return it->second;
        }
    }
    return std::nullopt;
}


} // namespace runtime
} // namespace metalchat
