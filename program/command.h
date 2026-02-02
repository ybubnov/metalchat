// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <bitset>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <argparse/argparse.hpp>

#include "config.h"
#include "manifest.h"


namespace metalchat {
namespace runtime {


struct ansi {
    static constexpr std::string_view yellow = "\e[0;33m";
    static constexpr std::string_view reset = "\e[0m";
};


using command_scope = unsigned long;


struct context_scope {
    static constexpr command_scope local = 1 << 0;
    static constexpr command_scope global = 1 << 1;
    static constexpr command_scope model = 1 << 2;

    static command_scope
    make_from_bool(bool is_local, bool is_global, bool is_model = false)
    {
        constexpr auto digits = std::numeric_limits<command_scope>::digits;
        std::bitset<digits> scope;

        scope |= local * static_cast<command_scope>(is_local);
        scope |= global * static_cast<command_scope>(is_global);
        scope |= model * static_cast<command_scope>(is_model);

        if (scope.count() == 0) {
            scope |= context_scope::local;
        }
        if (scope.count() > 1) {
            throw std::invalid_argument("error: only one scope at a time");
        }
        return scope.to_ulong();
    }

    static std::string
    string(command_scope flags)
    {
        static std::vector<std::string> names = {"local", "global", "model"};
        if (flags & local) {
            return "local";
        }
        if (flags & global) {
            return "global";
        }
        if (flags & model) {
            return "model";
        }
        return "undefined";
    }
};


struct command_context {
    using manifest_file = tomlfile<manifest>;

    std::filesystem::path root_path;
    tomlfile<config> config_file;
    std::unordered_map<command_scope, manifest_file> manifests;

    /// Returns a manifest file that corresponds to the requested command scope.
    ///
    /// Method optionally validates the presence of the manifest file using `missing_ok` flag.
    /// Also, when requested scope is empty, method defaults to the local scope.
    manifest_file
    resolve_manifest(command_scope scope, bool missing_ok = false) const
    {
        auto it = manifests.find(scope);
        if (it == manifests.end()) {
            std::runtime_error("fatal: requested non-existing scope");
        }
        auto file = it->second;
        if (!missing_ok && !file.exists()) {
            throw std::runtime_error("error: requested scope not checked out");
        }
        return file;
    }
};


class basic_command {
public:
    using parser_type = argparse::ArgumentParser;
    using parser_reference = std::reference_wrapper<parser_type>;
    using handler_type = std::function<void(const command_context&)>;

    basic_command(const std::string& name);
    basic_command(const std::string& name, const std::string& version);
    basic_command(const std::string& name, basic_command& parent);

    void
    push_handler(parser_type& parser, handler_type&& handler);

    void
    push_handler(basic_command& command);

    void
    handle(const command_context& context) const;

protected:
    /// Resolve the scope of the command by combininig flags `--local`, `--global`.
    ///
    /// The parser must define those flags explicitly, otherwise the method throws
    /// an exception.
    command_scope
    resolve_scope(const parser_type& parser) const;

    parser_type _M_command;

    template <typename Key, typename Value>
    using container_type = std::vector<std::pair<Key, Value>>;

    container_type<parser_reference, handler_type> _M_handlers;
};


} // namespace runtime
} // namespace metalchat
