// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <argparse/argparse.hpp>

#include "config.h"


namespace metalchat {
namespace runtime {


struct ansi {
    static constexpr std::string_view yellow = "\e[0;33m";
    static constexpr std::string_view reset = "\e[0m";
};


struct command_context {
    std::filesystem::path root_path;
    tomlfile<config> config_file;
};


class basic_command {
public:
    using parser_type = argparse::ArgumentParser;
    using parser_reference = std::reference_wrapper<parser_type>;
    using handler_type = std::function<void(const command_context&)>;

    basic_command(const std::string& name);
    basic_command(const std::string& name, basic_command& parent);

    void
    push_handler(parser_type& parser, handler_type&& handler);

    void
    push_handler(basic_command& command);

    void
    handle(const command_context& context) const;

protected:
    parser_type _M_command;

    template <typename Key, typename Value>
    using container_type = std::vector<std::pair<Key, Value>>;

    container_type<parser_reference, handler_type> _M_handlers;
};


} // namespace runtime
} // namespace metalchat
