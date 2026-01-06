// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "command.h"


namespace metalchat {
namespace program {


basic_command::basic_command(const std::string& name)
: _M_command(name),
  _M_handlers()
{}


basic_command::basic_command(const std::string& name, basic_command& parent)
: basic_command(name)
{
    parent.push_handler(_M_command, [&] { handle(); });
}


void
basic_command::push_handler(parser_type& parser, handler_type&& handler)
{
    _M_command.add_subparser(parser);
    _M_handlers.emplace_back(std::ref(parser), std::move(handler));
}


void
basic_command::push_handler(basic_command& command)
{
    push_handler(command._M_command, [&] { command.handle(); });
}


void
basic_command::handle() const
{
    for (const auto& [parser, handler] : _M_handlers) {
        if (_M_command.is_subcommand_used(parser)) {
            handler();
        }
    }
}


} // namespace program
} // namespace metalchat
