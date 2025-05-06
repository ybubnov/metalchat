#pragma once

#include <metalchat/bpe.h>
#include <metalchat/functional.h>
#include <metalchat/layer.h>
#include <metalchat/nn.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


class basic_message {
public:
    virtual std::string
    encode(const bpe&)
        = 0;
};


class system_message {};


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

    { transformer(input, std::size_t()) } -> is_future_tensor2_v<typename Transformer::value_type>;
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
        return top_p(logits.template flatten<2>(), _m_temperature, _m_p);
    }

    void
    print()
    {
        for (auto [name, _] : _m_estimator.get_parameters()) {
            std::cout << name << std::endl;
        }
    }

private:
    Estimator _m_estimator;
    hardware_accelerator _m_accelerator;
    value_type _m_temperature;
    value_type _m_p;
};


template <language_transformer_t Transformer> class chat {
public:
    chat(Transformer&& transformer)
    : _m_transformer(std::move(transformer))
    {}

    void
    send(const basic_message& message)
    {}

private:
    Transformer _m_transformer;
};


} // namespace metalchat
