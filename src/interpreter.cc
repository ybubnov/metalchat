// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <mstch/mstch.hpp>

#include <metalchat/interpreter.h>
#include <metalchat/transformer.h>

#include "metal_impl.h"


namespace mustache = mstch;


namespace metalchat {


namespace variables {


static const std::string command_format = R"(
To use a tool, respond with JSON in this format:
{"name":"command_name","parameters":{"param1":"value","param2":"value"}}
)";


} // namespace variables


struct interpreter::_Members {
    mustache::map vars;
    mustache::array commands;
    mustache::map builtins;

    _Members()
    : vars(),
      commands(),
      builtins({{"command_format", variables::command_format}, {"command", mustache::array()}})
    {}

    void
    push_command(const std::string& command)
    {
        commands.push_back(command);
        builtins.insert_or_assign("commands", commands);
        vars.insert_or_assign("metalchat", builtins);
    }

    const mustache::map&
    context() const
    {
        return vars;
    }

    std::string&
    at(const std::string& key)
    {
        return std::get<std::string>(vars[key]);
    }

    void
    assign(const std::string& key, const std::string& val)
    {
        vars.insert_or_assign(key, val);
    }
};


void
interpreter::construct()
{
    _M_members = std::make_shared<_Members>();

    // Do not escape characters, leave them as is. This is the global configuration,
    // so unfortunately this line changes behaviour for the whole library.
    mustache::config::escape = [](const std::string& str) -> std::string { return str; };
}


void
interpreter::declare_command(const std::string& declaration, command_type command)
{
    auto command_name = _M_command_scanner->declare(declaration);
    _M_commands.insert_or_assign(command_name, command);
    _M_members->push_command(declaration);
}


void
interpreter::declare_variable(const std::string& declaration, const std::string& value)
{
    _M_members->assign(declaration, value);
}


void
interpreter::write(const message_type& message)
{
    auto content = mustache::render(message.content(), _M_members->context());
    _M_format->format(message_type(message.role(), content), _M_stream);
}


} // namespace metalchat
