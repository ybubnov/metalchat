// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include "command.h"


namespace metalchat {
namespace runtime {


basic_command::basic_command(const std::string& name)
: _M_command(name),
  _M_handlers()
{}


basic_command::basic_command(const std::string& name, const std::string& version)
: _M_command(name, version),
  _M_handlers()
{}


basic_command::basic_command(const std::string& name, basic_command& parent)
: basic_command(name)
{
    parent.push_handler(_M_command, [&](const command_context& c) { handle(c); });
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
    push_handler(command._M_command, [&](const command_context& c) { command.handle(c); });
}


void
basic_command::handle(const command_context& context) const
{
    for (const auto& [parser, handler] : _M_handlers) {
        if (_M_command.is_subcommand_used(parser)) {
            handler(context);
        }
    }
}


command_scope
basic_command::resolve_scope(const parser_type& parser) const
{
    auto is_local = parser.is_used("--local");
    auto is_global = parser.is_used("--global");
    return context_scope::make_from_bool(is_local, is_global);
}


} // namespace runtime
} // namespace metalchat
