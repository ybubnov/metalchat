// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <fstream>
#include <istream>
#include <string_view>
#include <unordered_map>
#include <variant>

#include <metalchat/allocator.h>
#include <metalchat/nn.h>
#include <metalchat/safetensor.h>
#include <metalchat/tensor/concept.h>
#include <metalchat/text.h>


namespace metalchat {


/// A layer adaptor that is used by repository implementations to prepare the model for the usage.
///
/// Depending on the layer type and type of model distribution, the adaptor could be used to
/// (1) map weight names from one implementation to another, (2) rebuild a model to load
/// quantized implementation (QLoRa), etc.
template <typename Adaptor>
concept indirect_layer_adaptor = requires(std::remove_reference_t<Adaptor> const a) {
    /// An `adapt_pre` method is called, immediately after model creation and before loading
    /// weights from the file.
    { a.adapt_pre(nn::indirect_layer<nn::basic_layer>()) } -> std::same_as<void>;

    /// An `adapt_post` method is called after loading weights from file.
    { a.adapt_post(nn::indirect_layer<nn::basic_layer>()) } -> std::same_as<void>;
};


/// An implementation of \ref indirect_layer_adaptor concept that does nothing. Use it to declare
/// transformer traits (\ref transformer_traits) that do not perform any layer adaptation.
template <typename LayerOptions> struct noop_layer_adaptor {
    noop_layer_adaptor(LayerOptions) {}

    /// The implementation does not modify the specified layer.
    void
    adapt_pre(nn::indirect_layer<nn::basic_layer>) const
    {}

    /// The implementation does not modify the specified layer.
    void
    adapt_post(nn::indirect_layer<nn::basic_layer>) const
    {}
};


/// The stream serializer expects the type to load the `T` instance from the input stream,
/// and save the instance `T` to the output stream.
///
/// \tparam Serializer a type that implements methods `load` and `save`.
template <typename Serializer>
concept stream_serializer = requires(std::remove_reference_t<Serializer> const s) {
    typename Serializer::value_type;

    { s.load(std::declval<std::istream&>()) } -> std::same_as<typename Serializer::value_type>;
    {
        s.save(std::declval<std::ostream&>(), std::declval<typename Serializer::value_type&>())
    } -> std::same_as<void>;
};


/// The requirements for a transformer declaration.
template <typename Transformer>
concept language_transformer = requires {
    typename Transformer::layer_type;
    typename Transformer::layer_adaptor;
    typename Transformer::options_type;
    typename Transformer::options_serializer;
    typename Transformer::container_type;
    typename Transformer::document_adaptor;
    typename Transformer::tokenizer_type;
    typename Transformer::tokenizer_loader;

    requires nn::layer<typename Transformer::layer_type>;
    requires contiguous_container<typename Transformer::container_type>;
    requires indirect_layer_adaptor<typename Transformer::layer_adaptor>;
    requires safetensor_document_adaptor<typename Transformer::document_adaptor>;
    requires stream_serializer<typename Transformer::options_serializer>;
};


namespace detail {


template <typename Key, typename T> class mapping_iterator_wrapper {
public:
    using key_type = Key;
    using mapped_type = T;
    using value_type = std::pair<const Key, T>;
    using iterator_category = std::forward_iterator_tag;
    using iterator = mapping_iterator_wrapper<Key, T>;
    using reference = value_type&;
    using pointer = value_type*;
    using difference_type = std::ptrdiff_t;

    mapping_iterator_wrapper(std::unordered_map<Key, T>&& container)
    : _M_container(std::move(container)),
      _M_it(_M_container.begin())
    {}

    mapping_iterator_wrapper()
    : _M_container(),
      _M_it(_M_container.end())
    {}

    iterator&
    operator++()
    {
        ++_M_it;
        return *this;
    }

    reference
    operator*()
    {
        return *_M_it;
    }

    pointer
    operator->()
    {
        return &(*_M_it);
    }

    bool
    operator==(const iterator& rhs) const
    {
        return _M_it == rhs._M_it;
    }

    bool
    operator!=(const iterator& rhs) const
    {
        return _M_it != rhs._M_it;
    }

private:
    using container_type = std::unordered_map<Key, T>;
    using container_iterator = container_type::iterator;

    container_type _M_container;
    container_iterator _M_it;
};


class json_object {
public:
    using iterator = mapping_iterator_wrapper<std::string, std::string>;

    json_object(std::istream&);

    void
    merge(const std::string& key, bool value);

    void
    merge(const std::string& key, int value);

    void
    merge(const std::string& key, float value);

    void
    merge(const std::string& key, std::string&& value);

    void
    write(std::ostream&) const;

    iterator
    begin() const;

    iterator
    end() const;

private:
    struct _Members;
    std::shared_ptr<_Members> _M_members;
};


} // namespace detail


template <language_transformer Transformer> struct transformer_traits {
    using layer_type = Transformer::layer_type;
    using layer_adaptor_type = Transformer::layer_adaptor;
    using options_type = Transformer::options_type;
    using options_serializer = Transformer::options_serializer;
    using tokenizer_type = Transformer::tokenizer_type;
    using tokenizer_loader = Transformer::tokenizer_loader;
    using container_type = Transformer::container_type;
    using document_adaptor_type = Transformer::document_adaptor;

    /// Merge JSON-serializable options.
    ///
    /// This method uses JSON-query to replace a specified sequence of key-value pairs
    /// in the target options object, and then returns a new instance.
    ///
    /// Option keys must be specified as dot-separated path: `some.nested.value`.
    template <std::forward_iterator ForwardIt>
    static options_type
    merge_options(ForwardIt first, ForwardIt last, const options_type& options)
    {
        options_serializer serializer;

        std::stringstream input_stream;
        serializer.save(input_stream, options);

        detail::json_object object(input_stream);
        for (auto it = first; it != last; ++it) {
            auto& [key, value] = *it;
            std::visit([&](auto&& typed_value) {
                object.merge(key, std::move(typed_value));
            }, value);
        }

        std::stringstream output_stream;
        object.write(output_stream);

        return serializer.load(output_stream);
    }

    template <std::output_iterator<std::pair<std::string, std::string>> OutputIt>
    static void
    iter_options(const options_type& options, OutputIt output)
    {
        options_serializer serializer;

        std::stringstream input_stream;
        serializer.save(input_stream, options);

        detail::json_object object(input_stream);
        for (auto it = object.begin(); it != object.end(); ++it) {
            *output = *it;
            ++output;
        }
    }
};


/// Requirement of the `transformer_location` constant expression presence.
///
/// This concept is used in the repository implementation enabling methods for
/// retrieval of transformer instances from a default location within a repository.
template <typename T>
concept has_transformer_location = requires {
    T::transformer_location;

    requires std::same_as<decltype(T::transformer_location), std::string_view const>;
};

/// Requirement of the `options_location` constant expression presence.
///
/// This concept is used in the repository implementation enabling methods for
/// retrieval of option instances from a default location within a repository.
template <typename T>
concept has_options_location = requires {
    T::options_location;

    requires std::same_as<decltype(T::options_location), std::string_view const>;
};


/// Requirement of the `tokenizer_location` constant expression presence.
///
/// This concept is used in the repository implementation enabling methods for
/// retrieval of tokenizer instances from a default location within a repository.
template <typename T>
concept has_tokenizer_location = requires {
    T::tokenizer_location;

    requires std::same_as<decltype(T::tokenizer_location), std::string_view const>;
};


class basic_transformer {
public:
    using index_type = int32_t;
    using tensor_type = future_tensor<index_type, 2>;

    virtual tensor_type
    transform(tensor_type, std::size_t start_pos) = 0;

    virtual hardware_accelerator&
    accelerator() = 0;
};


template <typename Transformer> class transformer_wrapper : public basic_transformer {
public:
    transformer_wrapper(Transformer&& transformer)
    : _M_transformer(std::move(transformer))
    {}

    transformer_wrapper(const Transformer& transformer)
    : _M_transformer(transformer)
    {}

    tensor_type
    transform(tensor_type input, std::size_t start_pos)
    {
        return _M_transformer.transform(input, start_pos);
    }

    hardware_accelerator&
    accelerator()
    {
        return _M_transformer.get_layer().accelerator();
    }

private:
    Transformer _M_transformer;
};


template <typename Layer> class transformer {
public:
    using index_type = std::int32_t;
    using value_type = Layer::value_type;

    using layer_type = Layer;
    using layer_pointer = nn::indirect_layer<layer_type>;

    using sampler_type = nn::basic_sampler<value_type>;
    using sampler_pointer = std::shared_ptr<sampler_type>;

    transformer(const layer_pointer& layer, const sampler_pointer& sampler)
    : _M_layer(layer),
      _M_sampler(sampler)
    {}

    transformer(const layer_pointer& layer)
    : _M_layer(layer),
      _M_sampler(nn::make_default_sampler<value_type>())
    {}

    /// Replace current sampler with the specified implementation.
    ///
    /// \param sampler a new sampler instance to use in the transformation.
    void
    set_sampler(const sampler_pointer& sampler) noexcept
    {
        _M_sampler = sampler;
    }

    const sampler_pointer&
    get_sampler() const noexcept
    {
        return _M_sampler;
    }

    const layer_pointer&
    get_layer() const noexcept
    {
        return _M_layer;
    }

    layer_pointer&
    get_layer() noexcept
    {
        return _M_layer;
    }

    template <immutable_tensor2_t<index_type> Input>
    auto
    transform(Input input, std::size_t start_pos = 0)
    {
        auto& accelerator = _M_layer.accelerator();
        auto logits = _M_layer(input, start_pos).template flatten<2>();
        return _M_sampler->sample(logits, accelerator);
    }

private:
    nn::indirect_layer<layer_type> _M_layer;
    sampler_pointer _M_sampler;
};


} // namespace metalchat
