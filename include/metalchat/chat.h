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

    template <std::output_iterator<int32_t> OutputIt>
    void
    encode(const bpe& encoder, OutputIt output) const
    {
        encoder.encode(special_token::begin_header, output);
        encoder.encode(_m_role, output);
        encoder.encode(special_token::end_header, output);
        encoder.encode("\n\n", output);
        encoder.encode(_m_content, output);
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


/// The language estimator is an abstraction of the next token prediction model.
///
/// Types that comply to this concept are expected to produce logits for the all tokens
/// (characters, words, word combinations, etc.) that the given model is capable of
/// generating. All types should be inherited from `metalchat::layer` type to able assign
/// context of the model during runtime.
///
/// \verbatim embed:rst:leading-slashes
/// .. dropdown:: Concept determination
///    :icon: code
///
///    - A type declares indexing type `index_type`.
///    - A type declares value type `value_type`.
///    - A type declares input tensor type `input_tensor`: the first dimension is a batch
///      dimension, and the second dimension is a length of the input sequence.
///    - A type declare output tensor type `output_tensor`: the first dimension is a batch
///      dimension, the second dimension is a sequence length dimension, and the third dimension
///      is a length of model vocabulary.
///    - A type declares a method `estimate(input_tensor, std::size_t) -> output_tensor` that
///      implements the estimation logic and returns a tensor of logits.
/// \endverbatim
template <typename Estimator>
concept language_estimator_t = requires(Estimator estimator) {
    typename Estimator::index_type;
    typename Estimator::value_type;
    typename Estimator::input_tensor;
    typename Estimator::output_tensor;

    requires std::derived_from<Estimator, layer>;

    requires immutable_tensor2_t<typename Estimator::input_tensor, typename Estimator::index_type>;
    requires immutable_tensor3_t<typename Estimator::output_tensor, typename Estimator::value_type>;

    {
        estimator(std::declval<typename Estimator::input_tensor>(), std::size_t())
    } -> is_future_tensor3_v<typename Estimator::value_type>;
};


static_assert(language_estimator_t<nn::llama<float>>);
static_assert(language_estimator_t<nn::llama<dtype::bf16>>);


/// The language transformer is an abstraction of the next token generation model.
///
/// Types that comply to this concept are expected to produce index on the next token in the
/// model's vocabulary.
///
/// \verbatim embed:rst:leading-slashes
/// .. dropdown:: Concept determination
///    :icon: code
///
///    - A type declares indexing type `index_type`.
///    - A type declares an immutable 2-dimensional input tensor type `input_tensor`.
///    - A type declares an immutable 2-dimensional output tensor type `output_tensor`.
///    - A type declares a method `transform(input_tensor, std::size_t) -> output_tensor` that
///      implements a token transformation logic.
/// \endverbatim
template <typename Transformer>
concept language_transformer_t = requires(Transformer transformer) {
    typename Transformer::index_type;

    /// A transformer input tensor type, where the first dimension is a batch dimension, and the
    /// second dimension is a length of the input sequences.
    typename Transformer::input_tensor;

    /// A transformer output tensor type, where the first dimension is a batch dimension, and the
    /// second dimension is a length of the output sequence (a single token).
    typename Transformer::output_tensor;

    requires immutable_tensor2_t<
        typename Transformer::input_tensor, typename Transformer::index_type>;
    requires immutable_tensor2_t<
        typename Transformer::output_tensor, typename Transformer::index_type>;

    {
        transformer.transform(std::declval<typename Transformer::input_tensor>(), std::size_t())
    } -> is_future_tensor2_v<typename Transformer::index_type>;
};


class basic_language_transformer {
public:
    using index_type = int32_t;
    using input_tensor = future_tensor<index_type, 2>;
    using output_tensor = future_tensor<index_type, 2>;

    virtual output_tensor
    transform(input_tensor, std::size_t)
        = 0;

    virtual hardware_accelerator
    get_accelerator()
        = 0;
};


class polymorphic_language_transformer {
public:
    using index_type = int32_t;
    using input_tensor = future_tensor<index_type, 2>;
    using output_tensor = future_tensor<index_type, 2>;

    template <language_transformer_t Transformer>
    polymorphic_language_transformer(Transformer&& transformer)
    : _m_transformer(std::make_shared<Transformer>(std::move(transformer)))
    {}

    polymorphic_language_transformer(std::shared_ptr<basic_language_transformer> ptr);

    output_tensor
    transform(input_tensor, std::size_t);

    hardware_accelerator
    get_accelerator();

private:
    std::shared_ptr<basic_language_transformer> _m_transformer;
};


template <language_estimator_t Estimator>
class language_transformer : public basic_language_transformer {
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

    hardware_accelerator
    get_accelerator()
    {
        return _m_estimator.accelerator();
    }

    output_tensor
    transform(input_tensor input, std::size_t start_pos)
    {
        auto gpu = _m_estimator.accelerator();
        auto logits = _m_estimator(input, start_pos);
        return top_p(logits.template flatten<2>(), _m_temperature, _m_p, gpu);
    }

    template <immutable_tensor2_t<index_type> Input>
    output_tensor
    transform(Input input, std::size_t start_pos)
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

    void
    send(const basic_message& message)
    {
        auto output = std::back_inserter(_m_encoding);
        message.encode(_m_encoder, output);
        _m_encoder.encode(special_token::end_turn, output);
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
        query.encode(_m_encoder, std::back_inserter(_m_encoding));

        std::vector<index_type> encoding;
        _m_encoding.swap(encoding);

        auto encoding_size = encoding.size();
        auto container_ptr = std::make_shared<container_type>(std::move(encoding));

        auto alloc = _m_transformer.get_accelerator().get_allocator();
        auto input = future_tensor(tensor({1, encoding_size}, container_ptr), alloc);
        auto output = _m_transformer.transform(input, _m_start_pos);

        auto token = output.get()[0, 0];

        _m_start_pos += encoding_size;
        std::stringstream content;

        auto end_turn = _m_encoder.encode(special_token::end_turn);
        while (token != end_turn) {
            content << _m_encoder.decode(token);
            output = _m_transformer.transform(output, _m_start_pos++);
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


class polymorphic_chat {
public:
    using index_type = int32_t;
    using container_type = vector_memory_container<index_type>;

    template <language_transformer_t Transformer>
    polymorphic_chat(Transformer&& transformer, const bpe& encoder)
    : _m_chat(chat<polymorphic_language_transformer>(std::move(transformer), encoder))
    {}

    void
    send(const basic_message& message);

    std::string
    receive_text();

    basic_message
    receive();

private:
    chat<polymorphic_language_transformer> _m_chat;
};


template <language_transformer_t Transformer>
chat(Transformer&& transformer, const bpe& encoder) -> chat<Transformer>;


struct llama3_options {
public:
    llama3_options() {}
    llama3_options(const llama3_options&) = default;

    llama3_options
    head_dim(std::optional<std::size_t> head_dim) const noexcept;

    llama3_options
    n_heads(std::optional<std::size_t> n_heads) const noexcept;

    llama3_options
    n_kv_heads(std::optional<std::size_t> n_kv_heads) const noexcept;

    llama3_options
    n_layers(std::optional<std::size_t> n_layers) const noexcept;

    llama3_options
    max_seq_len(std::optional<std::size_t> max_seq_len) const noexcept;

    llama3_options
    heap_size(std::optional<std::size_t> heap_size) const noexcept;

    llama3_options
    rope_theta(std::optional<float> rope_theta) const noexcept;

    std::size_t
    head_dim() const noexcept;

    std::size_t
    n_heads() const noexcept;

    std::size_t
    n_kv_heads() const noexcept;

    std::size_t
    n_layers() const noexcept;

    std::size_t
    max_seq_len() const noexcept;

    std::size_t
    heap_size() const noexcept;

    float
    rope_theta() const noexcept;

private:
    std::size_t _m_head_dim = 0;
    std::size_t _m_n_heads = 0;
    std::size_t _m_n_kv_heads = 0;
    std::size_t _m_n_layers = 0;
    std::size_t _m_max_seq_len = 0;
    std::size_t _m_heap_size = 0;
    float _m_rope_theta = 0.0f;

    void
    set_head_dim(std::size_t head_dim);

    void
    set_n_heads(std::size_t n_heads);

    void
    set_n_kv_heads(std::size_t n_kv_heads);

    void
    set_n_layers(std::size_t n_layers);

    void
    set_max_seq_len(std::size_t max_seq_len);

    void
    set_heap_size(std::size_t heap_size);

    void
    set_rope_theta(float rope_theta);
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


using llama3_chat_type = llama3_traits<dtype::bf16>::type;


polymorphic_chat
make_llama3(
    const std::filesystem::path& weights_path,
    const std::filesystem::path& tokens_path,
    std::optional<llama3_options> options = std::nullopt
);


polymorphic_chat
make_llama3_compact(
    const std::filesystem::path& weights_path,
    const std::filesystem::path& tokens_path,
    std::optional<llama3_options> options = std::nullopt
);


polymorphic_chat
make_llama3(
    const std::string& weights_path,
    const std::string& tokens_path,
    std::optional<llama3_options> options = std::nullopt
);


polymorphic_chat
make_llama3_compact(
    const std::string& weights_path,
    const std::string& tokens_path,
    std::optional<llama3_options> options = std::nullopt
);


} // namespace metalchat
