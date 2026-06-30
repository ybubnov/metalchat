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


/// The stream serializer expects the type to load the `T` instance from the input stream,
/// and save the instance `T` to the output stream.
///
/// \tparam Serializer a type that implements methods `load` and `save`.
template <typename Serializer>
concept stream_serializer = requires(std::remove_reference_t<Serializer> const s) {
    typename Serializer::value_type;

    {
        /// Load the serializable value from the input stream.
        s.load(std::declval<std::istream&>())
    } -> std::same_as<typename Serializer::value_type>;
    {
        /// Save the serializable value into the output stream.
        s.save(std::declval<std::ostream&>(), std::declval<typename Serializer::value_type&>())
    } -> std::same_as<void>;
};


/// The requirements for a transformer declaration.
template <typename Transformer>
concept language_transformer = requires {
    typename Transformer::layer_type;
    typename Transformer::layer_serializer;
    typename Transformer::options_type;
    typename Transformer::options_serializer;
    typename Transformer::container_type;
    typename Transformer::tokenizer_type;
    typename Transformer::tokenizer_loader;

    requires nn::mutable_layer<typename Transformer::layer_type>;
    requires contiguous_container<typename Transformer::container_type>;
    requires safetensor_serializer<typename Transformer::layer_serializer>;
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
    using layer_serializer = Transformer::layer_serializer;
    using options_type = Transformer::options_type;
    using options_serializer = Transformer::options_serializer;
    using tokenizer_type = Transformer::tokenizer_type;
    using tokenizer_loader = Transformer::tokenizer_loader;
    using container_type = Transformer::container_type;

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

    virtual hardware_accelerator&
    accelerator() = 0;

    virtual tensor_type
    transform(tensor_type, std::size_t start_pos) = 0;
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
        return _M_transformer.accelerator();
    }

private:
    Transformer _M_transformer;
};


template <typename Layer, typename Index = std::int32_t> class transformer {
public:
    using index_type = Index;
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

    hardware_accelerator&
    accelerator()
    {
        return _M_layer.accelerator();
    }

    /// Replace current sampler with the specified implementation.
    ///
    /// When a null pointer is provided, then the sampler is set to an empty sequential
    /// sampler, which means that the \ref transform method returns all vocabulary indices.
    ///
    /// \param sampler a new sampler instance to use in the transformation.
    void
    set_sampler(const sampler_pointer& sampler) noexcept
    {
        if (sampler != nullptr) {
            _M_sampler = sampler;
        } else {
            using sampler_type = nn::sequential_sampler<value_type>;
            _M_sampler = std::make_shared<sampler_type>();
        }
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

    /// Transform the input sequence of tokens to the most probable token.
    ///
    /// \param input an input sequence of token identifiers (from a model vocabulary).
    /// \param start_pos a start position to generate a token from.
    template <immutable_tensor2_t<index_type> Input>
    auto
    transform(Input input, std::size_t start_pos = 0)
    {
        auto& accelerator = _M_layer.accelerator();
        auto logits = _M_layer(input, start_pos).template flatten<2>();
        return _M_sampler->sample(logits, accelerator);
    }

    template <immutable_tensor_t<index_type> Input>
    auto
    transform(Input input, std::size_t start_pos = 0)
    {
        return transform(input.expand_dims(0), start_pos);
    }

private:
    nn::indirect_layer<layer_type> _M_layer;
    sampler_pointer _M_sampler;
};


template <typename Layer, typename CharT, typename Traits = std::char_traits<CharT>>
class basic_layerbuf : public std::basic_streambuf<CharT, Traits> {
public:
    using streambuf_type = std::basic_streambuf<CharT, Traits>;
    using traits_type = streambuf_type::traits_type;
    using char_type = streambuf_type::char_type;
    using int_type = streambuf_type::int_type;
    using pos_type = streambuf_type::pos_type;
    using off_type = streambuf_type::off_type;

    using transformer_type = transformer<Layer>;
    using transformer_pointer = std::shared_ptr<transformer_type>;

    basic_layerbuf(const transformer_pointer& ptr, pos_type size, pos_type pos = 0)
    : streambuf_type(),
      _M_transformer(ptr),
      _M_pbuf(size),
      _M_gbuf(size),
      _M_queue(),
      _M_bufsize(size),
      _M_pos(pos)
    {
        setg(_M_gbuf.data(), _M_gbuf.data(), _M_gbuf.data() + _M_gbuf.size());

        // The subtract of 1 is necessary to guarantee that buffer fits into the
        // transformer cache, which is expected to be limited by a put buffer size.
        setp(_M_pbuf.data(), _M_pbuf.data() + _M_pbuf.size() - 1);
    }

    template <typename Layer>
    basic_layerbuf(const nn::indirect_layer<Layer>& layer, pos_type size, pos_type pos = 0)
    : basic_layerbuf(std::make_shared<transformer_type>(layer), size, pos)
    {}

protected:
    using tensor_type = future_tensor<char_type, 2>;
    using buffer_type = std::vector<char_type>;

    int_type
    underflow() override
    {
        /// The user is expected to provide at least a single input to make the
        /// stream readable. So return end-of-file, when the get token is empty.
        if (_M_queue.empty()) {
            return traits_type::eof();
        }

        auto ch = _M_queue.front().get()[0, 0];
        *gptr() = ch;
        gbump(1);

        auto token = _M_transformer.transform(_M_queue.front(), _M_pos++);

        _M_queue.pop();
        _M_queue.push(std::move(token));
        return traits_type::to_int_type(ch);
    }

    int_type
    overflow(int_type c = traits_type::eof()) override
    {
        char_type ch = traits_type::to_char_type(c);
        buffer_type output;

        auto size = pptr() - pbase();
        // Put area is empty and the character is set of the end of file,
        // there is nothing to do, simply return end of file.
        if (size == 0 && traits_type::eq_int_type(c, traits_type::eof())) {
            return traits_type::eof();
        }

        // This is safe, since the put buffer is exactly one character less, so overflow
        // happens on the buffer being smaller by a size of 1.
        if (!traits_type::eq_int_type(c, traits_type::eof())) {
            *pptr() = ch;
            size++;
        }

        buffer_type pbuf(_M_bufsize);
        _M_pbuf.swap(pbuf);
        setp(_M_pbuf.data(), _M_pbuf.data() + _M_pbuf.size() - 1);

        using container_type = vector_memory_container<CharT>;

        auto container_ptr = std::make_shared<container_type>(std::move(pbuf));
        auto pending = tensor({size}, container_ptr);

        auto token = _M_transformer.transform(pending, _M_pos);
        _M_queue.push(std::move(token));
        _M_pos += container_size;

        return traits_type::not_eof(c);
    }

    int
    sync() override
    {
        return overflow(traits_type::eof());
    }

private:
    transformer_pointer _M_transformer;
    buffer_type _M_pbuf;
    buffer_type _M_gbuf;
    std::queue<tensor_type> _M_queue;
    pos_type _M_bufsize;
    pos_type _M_pos;
};


} // namespace metalchat
