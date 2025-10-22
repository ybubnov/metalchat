#pragma once

#include <iostream>
#include <optional>

#include <metalchat/bpe.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/layer.h>
#include <metalchat/nn.h>
#include <metalchat/tensor.h>


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

    template <std::output_iterator<int32_t> OutputIt>
    void
    encode(const bpe& encoder, OutputIt output) const
    {
        encoder.encode(special_token::begin_header, output);
        encoder.encode(_M_role, output);
        encoder.encode(special_token::end_header, output);
        encoder.encode("\n\n", output);
        encoder.encode(_M_content, output);
    }

    std::string
    content() const
    {
        return _M_content;
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

    requires std::derived_from<Transformer, basic_layer>;

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


class agent {
public:
    using index_type = int32_t;
    using container_type = vector_memory_container<index_type>;

    template <typename Transformer>
    agent(Transformer&& transformer, const bpe& encoder)
    : _M_transformer(std::make_shared<transformer_wrapper<Transformer>>(std::move(transformer))),
      _M_encoder(encoder),
      _M_start_pos(0),
      _M_encoding(1, encoder.encode(special_token::begin_text))
    {}

    // void
    // knows_function(const func_spec&);

    // void
    // knows_function(std::function<void(func_spec&)> fn);

    void
    send(const basic_message& message)
    {
        auto output = std::back_inserter(_M_encoding);
        message.encode(_M_encoder, output);
        _M_encoder.encode(special_token::end_turn, output);
    }

    std::string
    receive_text()
    {
        return receive().content();
    }

    basic_message
    receive()
    {
        basic_message query("assistant");
        query.encode(_M_encoder, std::back_inserter(_M_encoding));

        std::vector<index_type> encoding;
        _M_encoding.swap(encoding);

        auto encoding_size = encoding.size();
        auto container_ptr = std::make_shared<container_type>(std::move(encoding));

        auto accelerator = _M_transformer->accelerator();
        auto alloc = accelerator.get_allocator();

        auto input = future_tensor(tensor({1, encoding_size}, container_ptr), alloc);
        auto output = _M_transformer->transform(input, _M_start_pos);

        auto token = output.get()[0, 0];

        _M_start_pos += encoding_size;
        std::stringstream content;

        auto end_turn = _M_encoder.encode(special_token::end_turn);
        while (token != end_turn) {
            content << _M_encoder.decode(token);
            output = _M_transformer->transform(output, _M_start_pos++);

            token = output.get()[0, 0];
        }

        return basic_message("assistant", content.str());
    }


private:
    std::shared_ptr<basic_transformer> _M_transformer;
    bpe _M_encoder;

    std::size_t _M_start_pos;
    std::vector<index_type> _M_encoding;
};


agent
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


// agent
// make_llama3_compact(
//     const std::string& weights_path,
//     const std::string& tokens_path,
//     std::optional<nn::llama3_options> options = std::nullopt
//);


} // namespace metalchat
