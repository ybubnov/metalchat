// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <argparse/argparse.hpp>


namespace metalchat {
namespace program {


class basic_command {
public:
    using parser_type = argparse::ArgumentParser;
    using parser_reference = std::reference_wrapper<parser_type>;
    using handler_type = std::function<void()>;

    basic_command(const std::string& name);
    basic_command(const std::string& name, basic_command& parent);

    void
    push_handler(parser_type& parser, handler_type&& handler);

    void
    push_handler(basic_command& command);

    void
    handle() const;

protected:
    parser_type _M_command;
    std::vector<std::pair<parser_reference, handler_type>> _M_handlers;
};


} // namespace program
} // namespace metalchat
