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

    chat(Transformer&& transformer, const bpe& encoder)
    : _m_transformer(std::move(transformer)),
      _m_encoder(encoder),
      _m_start_pos(0),
      _m_encoding(1, encoder.encode(special_token::begin_text))
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


struct llama3_options {
public:
    llama3_options() {}
    llama3_options(const llama3_options&) = default;

    llama3_options
    head_dim(std::optional<std::size_t> head_dim) const noexcept
    {
        llama3_options o = *this;
        if (head_dim.has_value()) {
            o.set_head_dim(head_dim.value());
        }
        return o;
    }

    llama3_options
    n_heads(std::optional<std::size_t> n_heads) const noexcept
    {
        llama3_options o = *this;
        if (n_heads.has_value()) {
            o.set_n_heads(n_heads.value());
        }
        return o;
    }

    llama3_options
    n_kv_heads(std::optional<std::size_t> n_kv_heads) const noexcept
    {
        llama3_options o = *this;
        if (n_kv_heads.has_value()) {
            o.set_n_kv_heads(n_kv_heads.value());
        }
        return o;
    }

    llama3_options
    n_layers(std::optional<std::size_t> n_layers) const noexcept
    {
        llama3_options o = *this;
        if (n_layers.has_value()) {
            o.set_n_layers(n_layers.value());
        }
        return o;
    }

    llama3_options
    max_seq_len(std::optional<std::size_t> max_seq_len) const noexcept
    {
        llama3_options o = *this;
        if (max_seq_len.has_value()) {
            o.set_max_seq_len(max_seq_len.value());
        }
        return o;
    }

    llama3_options
    heap_size(std::optional<std::size_t> heap_size) const noexcept
    {
        llama3_options o = *this;
        if (heap_size.has_value()) {
            o.set_heap_size(heap_size.value());
        }
        return o;
    }

    llama3_options
    rope_theta(std::optional<float> rope_theta) const noexcept
    {
        llama3_options o = *this;
        if (rope_theta.has_value()) {
            o.set_rope_theta(rope_theta.value());
        }
        return o;
    }


    std::size_t
    head_dim() const noexcept
    {
        return _m_head_dim;
    }

    std::size_t
    n_heads() const noexcept
    {
        return _m_n_heads;
    }

    std::size_t
    n_kv_heads() const noexcept
    {
        return _m_n_kv_heads;
    }

    std::size_t
    n_layers() const noexcept
    {
        return _m_n_layers;
    }

    std::size_t
    max_seq_len() const noexcept
    {
        return _m_max_seq_len;
    }

    std::size_t
    heap_size() const noexcept
    {
        return _m_heap_size;
    }

    float
    rope_theta() const noexcept
    {
        return _m_rope_theta;
    }

private:
    std::size_t _m_head_dim = 0;
    std::size_t _m_n_heads = 0;
    std::size_t _m_n_kv_heads = 0;
    std::size_t _m_n_layers = 0;
    std::size_t _m_max_seq_len = 0;
    std::size_t _m_heap_size = 0;
    float _m_rope_theta = 0.0f;

    void
    set_head_dim(std::size_t head_dim)
    {
        _m_head_dim = head_dim;
    }

    void
    set_n_heads(std::size_t n_heads)
    {
        _m_n_heads = n_heads;
    }

    void
    set_n_kv_heads(std::size_t n_kv_heads)
    {
        _m_n_kv_heads = n_kv_heads;
    }

    void
    set_n_layers(std::size_t n_layers)
    {
        _m_n_layers = n_layers;
    }

    void
    set_max_seq_len(std::size_t max_seq_len)
    {
        _m_max_seq_len = max_seq_len;
    }

    void
    set_heap_size(std::size_t heap_size)
    {
        _m_heap_size = heap_size;
    }

    void
    set_rope_theta(float rope_theta)
    {
        _m_rope_theta = rope_theta;
    }
};


llama3_options
default_llama3_1b_options();


template <typename T> struct llama3_traits {
    using value_type = T;
    using cache_type = nn::sink_cache<value_type>;
    using container_type = hardware_memory_container<value_type>;
    using estimator_type = nn::llama<value_type, container_type, cache_type>;
    using transformer_type = language_transformer<estimator_type>;
    using type = chat<transformer_type>;
};


llama3_traits<dtype::bf16>::type
construct_llama3_1b(
    const std::filesystem::path& weights_path,
    const std::filesystem::path& tokens_path,
    std::optional<llama3_options> options
);


} // namespace metalchat
