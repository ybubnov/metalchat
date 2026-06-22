// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <iostream>
#include <iterator>
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

private:
    std::string _M_role;
    std::string _M_content;
};


class basic_token_scanner {
public:
    using index_type = int32_t;

    virtual void
    reset() = 0;

    virtual bool
    scan(index_type token) = 0;

    /// The /ref basic_token_scanner default destructor.
    virtual ~basic_token_scanner() = default;
};


class match_token_scanner : public basic_token_scanner {
public:
    match_token_scanner(std::initializer_list<index_type> tokens)
    : match_token_scanner(tokens.begin(), tokens.end())
    {}

    template <std::forward_iterator ForwardIt>
    match_token_scanner(ForwardIt first, ForwardIt last)
        requires std::same_as<std::iter_value_t<ForwardIt>, index_type>
    : _M_tokens(first, last)
    {}

    void
    reset()
    {}

    bool
    scan(index_type token)
    {
        return _M_tokens.find(token) == _M_tokens.end();
    }

private:
    std::unordered_set<index_type> _M_tokens;
};


class limit_token_scanner : public basic_token_scanner {
public:
    limit_token_scanner(std::size_t lim)
    : _M_lim(lim),
      _M_scanned(0)
    {}

    void
    reset()
    {
        _M_scanned = 0;
    }

    bool
    scan(index_type token)
    {
        return (++_M_scanned) < _M_lim;
    }

private:
    std::size_t _M_lim;
    std::size_t _M_scanned;
};


template <typename LogicalOp> class composite_token_scanner : public basic_token_scanner {
public:
    using scanner_type = basic_token_scanner;
    using scanner_pointer = std::shared_ptr<scanner_type>;

    composite_token_scanner(std::initializer_list<scanner_pointer> scanners)
    : composite_token_scanner(scanners.begin(), scanners.end())
    {}

    template <std::forward_iterator ForwardIt>
    composite_token_scanner(ForwardIt first, ForwardIt last)
        requires std::same_as<std::iter_value_t<ForwardIt>, scanner_pointer>
    : _M_scanners(std::make_move_iterator(first), std::make_move_iterator(last)),
      _M_logical_op()
    {}

    /// The default \ref composite_token_scanner constructor.
    composite_token_scanner()
    : composite_token_scanner({})
    {}

    void
    reset()
    {
        for (auto& scanner : _M_scanners) {
            scanner->reset();
        }
    }

    bool
    scan(index_type token)
    {
        bool result = false;
        if (_M_scanners.size() == 0) {
            return result;
        }

        result = _M_scanners.front()->scan(token);
        for (std::size_t i = 1; i < _M_scanners.size(); i++) {
            result = _M_logical_op(result, _M_scanners[i]->scan(token));
        }
        return result;
    }

private:
    std::vector<scanner_pointer> _M_scanners;
    LogicalOp _M_logical_op;
};


/// Each message submitted to the interpreter is being passed through the mustache render engine,
/// so all valid mustache sequences are expanded with appropriate variable values.
class interpreter {
private:
    struct _Members;

    using index_type = int32_t;
    using tensor_type = future_tensor<index_type, 2>;
    using container_type = vector_memory_container<index_type>;

    using tokenizer_type = text::basic_tokenizer<index_type, char>;
    using tokenizer_traits = text::tokenizer_traits<tokenizer_type>;

    template <typename Transformer>
    static std::shared_ptr<basic_transformer>
    polymorphic_cast_transformer(const Transformer& transformer)
    {
        return std::make_shared<transformer_wrapper<Transformer>>(transformer);
    }

    template <typename Tokenizer>
    static std::shared_ptr<tokenizer_type>
    polymorphic_cast_tokenizer(const Tokenizer& tokenizer)
    {
        return std::make_shared<text::tokenizer_wrapper<Tokenizer>>(tokenizer);
    }

public:
    /// Type of the command handler used to process command calls.
    ///
    /// Interpreter executes a registered command, when an LLM model requests for an execution.
    using command_type = std::function<std::string(const command_statement&)>;

    template <typename Transformer, typename Tokenizer>
    interpreter(const Transformer& transformer, const Tokenizer& tokenizer)
    : interpreter(polymorphic_cast_transformer(transformer), polymorphic_cast_tokenizer(tokenizer))
    {}

    interpreter(
        std::shared_ptr<basic_transformer> transformer_ptr,
        std::shared_ptr<tokenizer_type> tokenizer_ptr
    );

    void
    set_token_scanner(std::shared_ptr<basic_token_scanner> scanner);

    template <std::derived_from<basic_token_scanner> Scanner>
    void
    set_token_scanner(Scanner&& scanner)
    {
        auto scanner_ptr = std::make_shared<Scanner>(std::move(scanner));
        set_token_scanner(scanner_ptr);
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
    read(OutputIt& output)
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
    std::shared_ptr<tokenizer_type> _M_tokenizer;
    std::shared_ptr<basic_token_scanner> _M_token_scanner;
    std::shared_ptr<basic_command_scanner> _M_command_scanner;
    std::unordered_map<std::string, command_type> _M_commands;

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
    read_until(OutputIt& it)
    {
        _M_token_scanner->reset();

        auto stream = flush();
        auto token = stream.get()[0, 0];

        while (_M_token_scanner->scan(token)) {
            tokenizer_traits::decode(*_M_tokenizer, token, it);
            stream = _M_transformer->transform(stream, _M_start_pos++);
            token = stream.get()[0, 0];
        }

        return stream;
    }
};


} // namespace metalchat
