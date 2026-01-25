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

    static command_scope
    make_from_bool(bool is_local, bool is_global)
    {
        command_scope scope = 0;
        scope |= (is_local ? local : 0);
        scope |= (is_global ? global : 0);
        return scope;
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
    resolve_manifest(command_scope flags, bool missing_ok = false) const
    {
        constexpr auto digits = std::numeric_limits<command_scope>::digits;
        std::bitset<digits> scope(flags);

        if (scope.count() == 0) {
            scope |= context_scope::local;
        }
        if (scope.count() > 1) {
            throw std::invalid_argument("error: only one manifest file at a time");
        }

        for (std::size_t i = 0; i < scope.size(); i++) {
            if (scope.test(i)) {
                auto it = manifests.find(command_scope(scope.to_ulong()));
                if (it == manifests.end()) {
                    std::runtime_error("fatal: requested non-existing scope");
                }

                auto file = it->second;
                if (!missing_ok && !file.exists()) {
                    throw std::runtime_error("error: requested scope not checked out");
                }
                return file;
            }
        }
        throw std::runtime_error("fatal: undefined command scope");
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
