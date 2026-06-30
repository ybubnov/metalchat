// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <iostream>
#include <iterator>
#include <optional>

#include <metalchat/command.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/functional.h>
#include <metalchat/nn.h>
#include <metalchat/tensor.h>
#include <metalchat/text.h>
#include <metalchat/transformer.h>


namespace metalchat {


/// Each message submitted to the interpreter is being passed through the mustache render engine,
/// so all valid mustache sequences are expanded with appropriate variable values.
class interpreter {
private:
    struct _Members;

    using index_type = int32_t;
    using char_type = char;

    using streambuf_type = std::basic_streambuf<index_type>;
    using stream_type = std::basic_iostream<index_type>;

public:
    /// Type of the command handler used to process command calls.
    ///
    /// Interpreter executes a registered command, when an LLM model requests for an execution.
    using command_type = std::function<std::string(const command_statement&)>;

    using formatter_type = basic_formatter<index_type, char_type>;
    using message_type = basic_message<char_type>;

    template <typename Layer>
    interpreter(transformer<Layer>&& t, formatter_type&& fmt)
    : _M_streambuf(nullptr),
      _M_stream(nullptr),
      _M_formatter(nullptr),
      _M_members(nullptr),
      _M_command_scanner(nullptr),
      _M_commands()
    {
        auto tptr = std::make_shared<transformer<Layer>>(std::move(t));

        _M_streambuf = std::make_shared<streambuf_type>(tptr);
        _M_iostream = std::make_shared<stream_type>(*_M_streambuf);
        _M_formatter = std::make_shared<formatter_type>(std::move(fmt));

        construct();
    }

    /// Declare the command available for execution.
    ///
    /// The declaration format depends on the underlying command scanner. By default command
    /// scanner is a \ref json_command_scanner, and declaration should be a
    /// [JSON Schema](https://json-schema.org/draft/2020-12) of the command and it's parameters.
    ///
    /// All command declarations are appended to the variable `{{metalchat_commands}}`.
    ///
    /// \param declaration A command declaration (i.e. JSON Schema by default).
    /// \param command A handler that returns the result of command execution.
    ///
    /// ```cpp
    /// auto command = R"({
    /// "name":"multiply",
    /// "type": "function",
    /// "description":"multiply two numbers",
    /// "parameters":{
    ///   "a":{"type":"number","description":"first number"},
    ///   "b":{"type":"number","description":"second number"}
    /// }})";
    ///
    /// interpreter interp(/* ... */);
    /// interp.declare_command(command, [](const command_statement&) -> std::string {
    ///    return R"({"result": nan})";
    /// });
    /// ```
    void
    declare_command(const std::string& declaration, command_type command);

    /// Declare the variable.
    ///
    /// The variable should not start with $-expansion symbol.
    ///
    /// ```cpp
    /// interpreter interp(/* ... */);
    /// interp.declare_variable("my_var", R"(arbitrary text)");
    /// ```
    ///
    /// \param declaration A variable name.
    /// \param value A variable value.
    void
    declare_variable(const std::string& declaration, const std::string& value);

    void
    write(const message_type& message);

    basic_message
    read()
    {
        return _M_formatter->parse(_M_stream);
    }

    basic_message
    exec()
    {
        for (;;) {
            auto message = _M_formatter->parse(_M_stream);
            if (message.role() != role::command) {
                return message;
            }

            auto command_statement = _M_command_scanner->scan(message.content());
            if (!command_statement) {
                return message;
            }

            auto& statement = command_statement.value();
            auto& command = _M_commands[statement.get_name()];
            auto response = message_type::response(command(statement));

            _M_formatter->format(response, _M_stream);
        }
    }

private:
    std::shared_ptr<streambuf_type> _M_streambuf;
    std::shared_ptr<stream_type> _M_stream;
    std::shared_ptr<formatter_type> _M_formatter;

    std::shared_ptr<_Members> _M_members;
    std::shared_ptr<basic_command_scanner> _M_command_scanner;
    std::unordered_map<std::string, command_type> _M_commands;

    void
    construct();
};


} // namespace metalchat
