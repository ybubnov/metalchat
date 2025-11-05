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
        encoder.encode(text::special_token::begin_header, output);
        encoder.encode(_M_role, output);
        encoder.encode(text::special_token::end_header, output);
        encoder.encode("\n\n", output);
        encoder.encode(_M_content, output);
    }

private:
    std::string _M_role;
    std::string _M_content;
};


/// The language transformer is an abstraction of the next token prediction model.
///
/// Types that comply to this concept are expected to produce logits for the all tokens
/// (characters, words, word combinations, etc.) that the given model is capable of
/// generating. All types should be inherited from `metalchat::basic_layer` type to able
/// assign context of the model during runtime.
template <typename Transformer>
concept transformer = requires(Transformer estimator) {
    typename Transformer::index_type;
    typename Transformer::value_type;
    typename Transformer::input_tensor;
    typename Transformer::output_tensor;

    requires std::derived_from<Transformer, nn::basic_layer>;

    requires immutable_tensor2_t<
        typename Transformer::input_tensor, typename Transformer::index_type>;
    requires immutable_tensor3_t<
        typename Transformer::output_tensor, typename Transformer::value_type>;

    {
        estimator(std::declval<typename Transformer::input_tensor>(), std::size_t())
    } -> is_future_tensor3_v<typename Transformer::value_type>;
};


class basic_transformer {
public:
    using index_type = int32_t;
    using tensor_type = future_tensor<index_type, 2>;

    virtual tensor_type
    transform(tensor_type, std::size_t start_pos)
        = 0;

    virtual hardware_accelerator&
    accelerator()
        = 0;
};


template <typename Transformer> class transformer_wrapper : public basic_transformer {
public:
    transformer_wrapper(Transformer&& transformer)
    : _M_transformer(std::move(transformer))
    {}

    tensor_type
    transform(tensor_type input, std::size_t start_pos)
    {
        return _M_transformer.transform(input, start_pos);
    }

    hardware_accelerator&
    accelerator()
    {
        return _M_transformer.accelerator();
    }

private:
    Transformer _M_transformer;
};


class interpreter {
private:
    using index_type = int32_t;
    using tensor_type = future_tensor<index_type, 2>;
    using container_type = vector_memory_container<index_type>;

public:
    using command_type = std::function<std::string(const command_statement&)>;

    template <typename Transformer>
    interpreter(Transformer&& transformer, const text::bpe& encoder, std::size_t max_pos = -1)
    : _M_transformer(std::make_shared<transformer_wrapper<Transformer>>(std::move(transformer))),
      _M_command_scanner(std::make_shared<json_command_scanner>()),
      _M_commands(),
      _M_encoder(encoder),
      _M_max_pos(max_pos),
      _M_start_pos(0),
      _M_buf(1, encoder.encode(text::special_token::begin_text))
    {}

    void
    register_command(const std::string& declaration, command_type command)
    {
        auto command_name = _M_command_scanner->declare(declaration);
        _M_commands.insert_or_assign(command_name, command);
    }

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
        write_header("assistant");

        std::stringstream content;
        std::ostream_iterator<std::string> content_iterator(content);

        read(content_iterator, text::special_token::end_turn);
        return basic_message("assistant", content.str());
    }

    basic_message
    exec()
    {
        basic_message message("assistant");

        do {
            write_header("assistant");

            std::stringstream content;
            std::ostream_iterator<std::string> content_iterator(content);

            read(content_iterator, text::special_token::end_turn);
            message = basic_message("assistant", content.str());

            auto command_statement = _M_command_scanner->scan(content.str());
            if (command_statement.has_value()) {
                std::cout << "COMMAND=<<<" << content.str() << ">>>" << std::endl;
                auto& statement = command_statement.value();
                auto& command = _M_commands[statement.get_name()];
                auto command_output = command(statement);

                write(basic_message("python", command_output));
            }

        } while (!_M_buf.empty());

        return message;
    }

private:
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
        for (auto& e : _M_buf) {
            std::cout << _M_encoder.decode(e);
        }
        std::cout << std::flush;
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
    read(OutputIt it, text::special_token special_token)
    {
        auto stream = flush();

        auto token = stream.get()[0, 0];
        auto util_token = _M_encoder.encode(special_token);

        while (token != util_token) {
            *it++ = _M_encoder.decode(token);
            stream = _M_transformer->transform(stream, _M_start_pos++);
            token = stream.get()[0, 0];
        }

        return stream;
    }
};


interpreter
make_llama3(
    const std::filesystem::path& weights_path,
    const std::filesystem::path& tokens_path,
    std::optional<nn::llama3_options> options = std::nullopt
);


// agent
// make_llama3_compact(
//     const std::filesystem::path& weights_path,
//     const std::filesystem::path& tokens_path,
//     std::optional<nn::llama3_options> options = std::nullopt
//);


} // namespace metalchat
