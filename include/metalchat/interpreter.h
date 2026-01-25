// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <iostream>
#include <optional>

#include <metalchat/command.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/nn.h>
#include <metalchat/tensor.h>
#include <metalchat/text.h>
#include <metalchat/transformer.h>


namespace metalchat {


class basic_message {
public:
    basic_message(const std::string& role, const std::string& content)
    : _M_role(role),
      _M_content(content)
    {}

    basic_message(std::string&& role, std::string&& content)
    : _M_role(std::move(role)),
      _M_content(std::move(content))
    {}

    basic_message(const std::string& role)
    : _M_role(role),
      _M_content()
    {}

    basic_message(const basic_message&) = default;

    const std::string&
    role() const
    {
        return _M_role;
    }

    const std::string&
    content() const
    {
        return _M_content;
    }

    template <std::output_iterator<int32_t> OutputIt>
    void
    encode(const text::bpe& encoder, OutputIt output) const
    {
        encoder.encode(text::token::begin_header, output);
        encoder.encode(_M_role, output);
        encoder.encode(text::token::end_header, output);
        encoder.encode("\n\n", output);
        encoder.encode(_M_content, output);
    }

private:
    std::string _M_role;
    std::string _M_content;
};


class interpreter {
private:
    struct _Members;

    using index_type = int32_t;
    using tensor_type = future_tensor<index_type, 2>;
    using container_type = vector_memory_container<index_type>;

    template <typename Transformer>
    static std::shared_ptr<basic_transformer>
    wrap(Transformer&& transformer)
    {
        return std::make_shared<transformer_wrapper<Transformer>>(std::move(transformer));
    }

public:
    /// Type of the command handler used to process command calls.
    ///
    /// Interpreter executes a registered command, when an LLM model requests for an execution.
    using command_type = std::function<std::string(const command_statement&)>;

    /// The structure keeps variable names accessing to the interpreter.
    ///
    /// Each message submitted to the interpreter is being passed through the mustache
    /// render engine, so all valid mustache sequences are expanded with appropriate
    /// variable values.
    struct variable {
        /// Variable `metalchat_commands`
        static const std::string commands;

        /// Variable `metalchat_command_format`
        static const std::string command_format;
    };

    template <typename Transformer>
    interpreter(Transformer&& transformer, const text::bpe& encoder, std::size_t max_pos = -1)
    : interpreter(wrap(std::move(transformer)), encoder, max_pos)
    {}

    template <typename Transformer>
    interpreter(const Transformer& transformer, const text::bpe& encoder, std::size_t max_pos = -1)
    : interpreter(wrap(transformer), encoder, max_pos)
    {}

    interpreter(
        std::shared_ptr<basic_transformer> transformer_ptr,
        const text::bpe& encoder,
        std::size_t max_pos = -1
    );

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
    /// ```c++
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
    /// ```c++
    /// interpreter interp(/* ... */);
    /// interp.declare_variable("my_var", R"(arbitrary text)");
    /// ```
    ///
    /// \param declaration A variable name.
    /// \param value A variable value.
    void
    declare_variable(const std::string& declaration, const std::string& value);

    // void
    // reset();

    void
    write(const basic_message& message);

    std::string
    read_text()
    {
        return read().content();
    }

    basic_message
    read()
    {
        std::stringstream content;
        std::ostream_iterator<std::string> content_iterator(content);

        read(content_iterator);
        return basic_message("assistant", content.str());
    }

    template <std::output_iterator<std::string> OutputIt>
    void
    read(OutputIt output)
    {
        write_header("assistant");
        read_until(output);
    }

    basic_message
    exec()
    {
        basic_message message("assistant");

        do {
            message = read();

            auto command_statement = _M_command_scanner->scan(message.content());
            if (command_statement.has_value()) {
                auto& statement = command_statement.value();
                auto& command = _M_commands[statement.get_name()];
                auto command_output = command(statement);

                write(basic_message("ipython", command_output));
            }

        } while (!_M_buf.empty());

        return message;
    }

private:
    std::shared_ptr<_Members> _M_members;
    std::shared_ptr<basic_transformer> _M_transformer;
    std::shared_ptr<basic_command_scanner> _M_command_scanner;
    std::unordered_map<std::string, command_type> _M_commands;
    text::bpe _M_encoder;

    std::size_t _M_max_pos;
    std::size_t _M_start_pos;
    std::vector<index_type> _M_buf;

    void
    write_header(const std::string& role);

    tensor_type
    flush()
    {
        std::vector<index_type> encoding;
        _M_buf.swap(encoding);

        auto encoding_size = encoding.size();
        auto container_ptr = std::make_shared<container_type>(std::move(encoding));

        auto accelerator = _M_transformer->accelerator();
        auto alloc = accelerator.get_allocator();

        auto input = future_tensor(tensor({1, encoding_size}, container_ptr), alloc);
        auto stream = _M_transformer->transform(input, _M_start_pos);

        _M_start_pos += encoding_size;
        return stream;
    }

    template <std::output_iterator<std::string> OutputIt>
    tensor_type
    read_until(OutputIt it)
    {
        auto stream = flush();
        auto token = stream.get()[0, 0];

        auto end_turn = _M_encoder.encode(text::token::end_turn);
        auto end_message = _M_encoder.encode(text::token::end_message);

        while (token != end_turn && token != end_message) {
            *it++ = _M_encoder.decode(token);
            stream = _M_transformer->transform(stream, _M_start_pos++);
            token = stream.get()[0, 0];
        }

        return stream;
    }
};


} // namespace metalchat
