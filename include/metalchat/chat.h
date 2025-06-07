#pragma once

#include <metalchat/bpe.h>
#include <metalchat/functional.h>
#include <metalchat/layer.h>
#include <metalchat/nn.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename Encodable, typename PushBackContainer>
concept __byte_pair_encodable = requires(const Encodable encodable, PushBackContainer& container) {
    requires push_back_container<PushBackContainer>;

    { encodable.encode(std::declval<const bpe&>(), container) } -> std::same_as<void>;
};


template <typename Encodable>
concept byte_pair_encodable = __byte_pair_encodable<Encodable, std::vector<bpe::index_type>>;


class basic_message {
public:
    basic_message(const std::string& role, const std::string& content)
    : _m_role(role),
      _m_content(content)
    {}

    basic_message(std::string&& role, std::string&& content)
    : _m_role(std::move(role)),
      _m_content(std::move(content))
    {}

    basic_message(const std::string& role)
    : _m_role(role),
      _m_content()
    {}

    template <push_back_container PushBackContainer>
    void
    encode(const bpe& encoder, PushBackContainer& container) const
    {
        encoder.encode(special_token::begin_header, container);
        encoder.encode(_m_role, container);
        encoder.encode(special_token::end_header, container);
        encoder.encode("\n\n", container);
        encoder.encode(_m_content, container);
    }

    std::string
    content() const
    {
        return _m_content;
    }

private:
    std::string _m_role;
    std::string _m_content;
};


template <typename Input, typename Estimator>
concept __language_estimator_t = requires(Input input, Estimator estimator) {
    requires std::derived_from<Estimator, layer>;
    requires immutable_tensor2_t<Input, int32_t>;

    typename Estimator::value_type;

    { estimator(input, std::size_t()) } -> is_future_tensor3_v<typename Estimator::value_type>;
};


/// The language estimator is an abstraction of the next token prediction model.
///
/// Types that comply to this concept are expected to produce logits for the all tokens
/// (characters, words, word combinations, etc.) that the given model is capable of generating.
/// All types should be inherited from `layer` type to able assign context of the model during
/// runtime.
///
/// The `Input` parameter of this concept is a tensor, where the first dimension is a batch
/// dimension, and the second dimension is a length of the input sequence.
template <typename Estimator>
concept language_estimator_t = __language_estimator_t<future_tensor<int32_t, 2>, Estimator>;

static_assert(language_estimator_t<nn::llama<float>>);
static_assert(language_estimator_t<nn::llama<dtype::bf16>>);


template <typename Input, typename Transformer>
concept __language_transformer_t = requires(Input input, Transformer transformer) {
    requires immutable_tensor2_t<Input, int32_t>;

    typename Transformer::value_type;

    { transformer(input, std::size_t()) } -> is_future_tensor2_v<int32_t>;
};


template <typename Transformer>
concept language_transformer_t = __language_transformer_t<future_tensor<int32_t, 2>, Transformer>;


template <language_estimator_t Estimator> class language_transformer {
public:
    using value_type = Estimator::value_type;

    language_transformer(const language_transformer&) = delete;
    language_transformer(language_transformer&&) = default;

    /// TODO: Introduce sampling classes for the model sampling configuration.
    language_transformer(
        Estimator&& estimator,
        value_type temperature = value_type(0.6),
        value_type p = value_type(0.9)
    )
    : _m_estimator(std::move(estimator)),
      _m_temperature(temperature),
      _m_p(p)
    {}

    future_tensor<int32_t, 2>
    operator()(future_tensor<int32_t, 2> input, std::size_t start_pos)
    {
        auto gpu = _m_estimator.accelerator();
        auto logits = _m_estimator(input, start_pos);
        return top_p(logits.template flatten<2>(), _m_temperature, _m_p, gpu);
    }

    template <immutable_tensor2_t<int32_t> Input>
    future_tensor<int32_t, 2>
    operator()(Input input, std::size_t start_pos)
    {
        auto gpu = _m_estimator.accelerator();
        auto logits = _m_estimator(input, start_pos);
        return top_p(logits.template flatten<2>(), _m_temperature, _m_p, gpu);
    }

private:
    Estimator _m_estimator;
    value_type _m_temperature;
    value_type _m_p;
};


template <language_transformer_t Transformer> class chat {
public:
    using index_type = int32_t;
    using container_type = vector_memory_container<index_type>;

    chat(Transformer&& transformer, bpe&& encoder)
    : _m_transformer(std::move(transformer)),
      _m_encoder(std::move(encoder)),
      _m_encoding(encoder.encode(special_token::begin_text))
    {}

    template <byte_pair_encodable Message>
    void
    send(const Message& message)
    {
        message.encode(_m_encoder, _m_encoding);
        _m_encoder.encode(special_token::end_turn, _m_encoding);
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
        query.encode(_m_encoder, _m_encoding);

        std::vector<index_type> encoding;
        _m_encoding.swap(encoding);

        auto encoding_size = encoding.size();
        auto container_ptr = std::make_shared<container_type>(std::move(encoding));

        auto input = tensor({1, encoding_size}, container_ptr);
        auto output = _m_transformer(shared_tensor(std::move(input)), _m_start_pos);
        auto token = output.get()[0, 0];

        _m_start_pos += encoding_size;
        std::stringstream content;

        auto end_turn = _m_encoder.encode(special_token::end_turn);
        while (token != end_turn) {
            content << _m_encoder.decode(token);
            output = _m_transformer(output, _m_start_pos++);
            token = output.get()[0, 0];
        }

        return basic_message("assistant", content.str());
    }


private:
    Transformer _m_transformer;
    bpe _m_encoder;

    std::size_t _m_start_pos;
    std::vector<index_type> _m_encoding;
};


template <language_transformer_t Transformer>
chat(Transformer&& transformer, const bpe& encoder) -> chat<Transformer>;


template <typename T>
auto
make_chat(const std::filesystem::path& weights_path, const std::filesystem::path& tokens_path)
{
    metalchat::hardware_accelerator gpu0;

    metalchat::bpe bpe(tokens_path);
    metalchat::safetensor_file weights(weights_path);

    auto alloc1 = hardware_nocopy_allocator(gpu0.get_allocator(), gpu0.get_metal_device());
    auto alloc2 = hardware_resident_allocator(alloc1, gpu0.get_metal_device());

    gpu0.set_allocator(std::move(alloc2));
    auto options = nn::attention_options{
        .head_dim = 64,
        .n_heads = 32,
        .n_kv_heads = 8,
        .max_seq_len = 32,
        .rope_theta = 500000.0
    };

    nn::llama<T> m(16, options, gpu0);
    m.initialize(weights, make_rebind_allocator<T>(gpu0.get_allocator()));

    auto transformer = language_transformer(std::move(m));
    auto agent = chat(std::move(transformer), std::move(bpe));

    agent.send(basic_message("system", "You are a helpful assistant"));
    agent.send(basic_message("user", "What is the capital of France?"));
    std::cout << agent.receive_text() << std::endl;
    // return std::make_tuple(std::move(agent), std::move(weights));
}


} // namespace metalchat
