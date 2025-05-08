#pragma once

#include <metalchat/bpe.h>
#include <metalchat/functional.h>
#include <metalchat/layer.h>
#include <metalchat/nn.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename Message, typename PushBackContainer>
concept language_encodable_message = requires(const Message message, PushBackContainer& container) {
    { message.encode(std::declval<byte_pair_encoder&>(), container) } -> std::same_as<void>;
} && push_back_container<PushBackContainer>;


class system_message {
public:
    system_message(const std::string& text)
    : _m_text(text)
    {}

    template <push_back_container PushBackContainer>
    void
    encode(byte_pair_encoder& bpe, PushBackContainer& container) const
    {
        bpe.encode(special_token::begin_header, container);
        bpe.encode("system", container);
        bpe.encode(special_token::end_header, container);
        bpe.encode("\n\n", container);
        bpe.encode(_m_text, container);
        bpe.encode(special_token::end_turn, container);
    }

private:
    std::string _m_text;
};


class user_message {};


class assistant_message {};


template <typename Input, typename Estimator>
concept __language_estimator_t = requires(Input input, Estimator estimator) {
    typename Estimator::value_type;

    { estimator(input, std::size_t()) } -> is_future_tensor3_v<typename Estimator::value_type>;
} && immutable_tensor2_t<Input, int32_t> && std::derived_from<Estimator, layer>;


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
    typename Transformer::value_type;

    { transformer(input, std::size_t()) } -> is_future_tensor2_v<int32_t>;
} && immutable_tensor2_t<Input, int32_t>;


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
        hardware_accelerator accelerator,
        value_type temperature = value_type(0.6),
        value_type p = value_type(0.9)
    )
    : _m_estimator(std::move(estimator)),
      _m_accelerator(accelerator),
      _m_temperature(temperature),
      _m_p(p)
    {}

    future_tensor<int32_t, 2>
    operator()(future_tensor<int32_t, 2> input, std::size_t start_pos)
    {
        auto logits = _m_estimator(input, start_pos);
        return top_p(logits.template flatten<2>(), _m_temperature, _m_p, _m_accelerator);
    }

    template <immutable_tensor2_t<int32_t> Input>
    future_tensor<int32_t, 2>
    operator()(Input input, std::size_t start_pos)
    {
        auto logits = _m_estimator(input, start_pos);
        return top_p(logits.template flatten<2>(), _m_temperature, _m_p, _m_accelerator);
    }

private:
    Estimator _m_estimator;
    hardware_accelerator _m_accelerator;
    value_type _m_temperature;
    value_type _m_p;
};


template <language_transformer_t Transformer> class chat {
public:
    chat(Transformer&& transformer, byte_pair_encoder&& bpe)
    : _m_transformer(std::move(transformer)),
      _m_bpe(std::move(bpe))
    {}

    template <language_encodable_message<std::vector<int32_t>> Message>
    std::string
    send(const Message& message, std::size_t max_size = 20)
    {
        auto encoding = std::vector<int32_t>();

        _m_bpe.encode(special_token::begin_text, encoding);
        message.encode(_m_bpe, encoding);

        _m_bpe.encode(special_token::begin_header, encoding);
        _m_bpe.encode("user", encoding);
        _m_bpe.encode(special_token::end_header, encoding);
        _m_bpe.encode("\n\n", encoding);
        _m_bpe.encode("What is the capital of France?", encoding);
        _m_bpe.encode(special_token::end_turn, encoding);

        _m_bpe.encode(special_token::begin_header, encoding);
        _m_bpe.encode("assistant", encoding);
        _m_bpe.encode(special_token::end_header, encoding);
        _m_bpe.encode("\n\n", encoding);
        _m_bpe.encode(special_token::begin_text, encoding);

        auto encoding_size = encoding.size();

        auto msg = _m_bpe.decode(encoding.begin(), encoding.end());
        std::cout << msg;

        auto container_ptr
            = std::make_shared<vector_memory_container<int32_t>>(std::move(encoding));
        auto input = tensor<int32_t, 2, vector_memory_container<int32_t>>(
            {1, encoding_size}, container_ptr
        );
        auto output = _m_transformer(shared_tensor(std::move(input)), 0);
        std::cout << _m_bpe.decode(output.get()[0, 0]);

        std::stringstream ss;
        for (std::size_t i = encoding_size; i < encoding_size + max_size - 1; i++) {
            output = _m_transformer(output, i);
            std::cout << _m_bpe.decode(output.get()[0, 0]);
        }
        std::cout << std::endl;
        return ss.str();
    }

private:
    Transformer _m_transformer;
    byte_pair_encoder _m_bpe;
};


template <language_transformer_t Transformer>
chat(Transformer&& transformer, byte_pair_encoder& bpe) -> chat<Transformer>;


} // namespace metalchat
