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
concept language_estimator_t
    = requires(Estimator estimator) { typename Estimator::result_type; }
      && std::derived_from<Estimator, layer>
      && is_future_tensor_t<typename Estimator::result_type, 3>
      && std::is_invocable_r<
          typename Estimator::result_type,
          Estimator,
          shared_hardware_tensor<typename Estimator::value_type, 2>>::value;


static_assert(language_estimator_t<nn::llama<float, hardware_memory_container<float>>>);


template <typename Transformer>
concept language_transformer_t = requires(std::remove_reference_t<Transformer> transformer) {
    //{ transformer(std::declval<Input>(), std::declval<std::size_t>()) } ->
    //    std::same_as<future_tensor<int32_t, 2>>;
    typename Transformer::value_type;
};


template <language_estimator_t Estimator> class llama_transformer {
public:
    using value_type = Estimator::value_type;

    llama_transformer(const llama_transformer&) = delete;
    llama_transformer(llama_transformer&&) = default;

    /// TODO: Introduce sampling classes for the model sampling configuration.
    llama_transformer(
        Estimator&& estimator,
        hardware_accelerator& gpu,
        value_type temp = value_type(0.6),
        value_type p = value_type(0.9)
    )
    : _m_estimator(std::move(estimator)),
      _m_gpu(gpu),
      _m_temp(temp),
      _m_p(p)
    {}

    template <immutable_tensor2_t<int32_t> Input>
    future_tensor<int32_t, 2>
    operator()(Input input, std::size_t start_pos)
    {
        auto logits = _m_estimator(input, start_pos);
        return top_p(logits.template flatten<2>(), _m_temp, _m_p);
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
    hardware_accelerator& _m_gpu;
    value_type _m_temp;
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
