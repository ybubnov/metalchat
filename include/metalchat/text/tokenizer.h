// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <metalchat/container.h>
#include <metalchat/tensor.h>


namespace metalchat {
namespace text {


/// Specifies kind of the token.
///
/// Tokens are used to transform a natural language sentences into a vector of integers
/// mapping them to a embedding space of the respective language model. There are specific
/// kinds of tokens that allow to instruct the model for a specific behaviour.
using tokenkind = int32_t;


struct token {
    static constexpr tokenkind regular = 1 << 0;
    static constexpr tokenkind begin_text = 1 << 1;
    static constexpr tokenkind end_text = 1 << 2;
    static constexpr tokenkind reserved = 1 << 3;
    static constexpr tokenkind finetune_right_pad = 1 << 4;
    static constexpr tokenkind begin_header = 1 << 5;
    static constexpr tokenkind end_header = 1 << 6;
    static constexpr tokenkind end_message = 1 << 7;
    static constexpr tokenkind end_turn = 1 << 8;
    static constexpr tokenkind ipython = 1 << 9;
};

template <typename T> struct basic_output_iterator {
    using iterator_category = std::output_iterator_tag;
    using value_type = void;
    using pointer = void;
    using reference = void;
    using difference_type = std::ptrdiff_t;

    virtual basic_output_iterator&
    operator++()
    {
        return *this;
    };

    basic_output_iterator
    operator++(int)
    {
        throw std::runtime_error("basic_output_iterator: no use");
    }

    virtual basic_output_iterator&
    operator*()
    {
        return *this;
    };

    virtual basic_output_iterator&
    operator=(const T&)
    {
        return *this;
    };

    virtual basic_output_iterator&
    operator=(T&&)
    {
        return *this;
    };

    virtual ~basic_output_iterator() = default;
};


template <typename T, std::output_iterator<T> OutputIt>
struct output_iterator_wrapper : public basic_output_iterator<T> {
private:
    OutputIt* _M_it;

public:
    using value_type = T;
    using basic_type = basic_output_iterator<value_type>;

    output_iterator_wrapper(OutputIt& output)
    : _M_it(std::addressof(output))
    {}

    output_iterator_wrapper(OutputIt* output_ptr)
    : _M_it(output_ptr)
    {}

    basic_type&
    operator++() override
    {
        ++(*_M_it);
        return *this;
    }

    basic_type&
    operator*() override
    {
        return *this;
    }

    basic_type&
    operator=(const value_type& value) override
    {
        **_M_it = value;
        return *this;
    }

    basic_type&
    operator=(value_type&& value) override
    {
        **_M_it = std::move(value);
        return *this;
    }
};


template <typename Index, typename CharT> struct basic_tokenizer {
    using index_type = Index;
    using string_type = std::basic_string<CharT>;
    using encoding_iterator = basic_output_iterator<index_type>;
    using decoding_iterator = basic_output_iterator<string_type>;

    virtual void
    encode(tokenkind kind, encoding_iterator& output) const = 0;

    virtual void
    encode(const string_type& s, encoding_iterator& output) const = 0;

    virtual void
    decode(index_type id, decoding_iterator& output) const = 0;

    /// The \ref basic_tokenizer default destructor.
    virtual ~basic_tokenizer() = default;
};


template <typename I, typename T>
concept forward_iterator = std::forward_iterator<I> && std::same_as<std::iter_value_t<I>, T>;


template <typename Tokenizer> struct tokenizer_traits {
    using index_type = Tokenizer::index_type;
    using string_type = Tokenizer::string_type;
    using char_type = string_type::value_type;
    using encoding_iterator = Tokenizer::encoding_iterator;
    using decoding_iterator = Tokenizer::decoding_iterator;

    template <std::output_iterator<index_type> OutputIt>
    static void
    encode(const Tokenizer& t, const string_type& s, OutputIt& output)
    {
        using iterator_wrapper = output_iterator_wrapper<index_type, OutputIt>;
        iterator_wrapper output_it(output);
        t.encode(s, output_it);
    }

    template <std::output_iterator<index_type> OutputIt>
    static void
    encode(const Tokenizer& t, tokenkind kind, OutputIt& output)
    {
        using iterator_wrapper = output_iterator_wrapper<index_type, OutputIt>;
        iterator_wrapper output_it(output);
        t.encode(kind, output_it);
    }

    static index_type
    encode(const Tokenizer& t, tokenkind kind)
    {
        using iterator = index_type*;
        using iterator_wrapper = output_iterator_wrapper<index_type, iterator>;

        index_type id;
        index_type* id_ptr = &id;
        iterator_wrapper output_it(id_ptr);

        t.encode(kind, output_it);
        return id;
    }

    static auto
    encode(const Tokenizer& t, const string_type& s)
    {
        using container_type = vector_memory_container<index_type>;
        using storage_type = container_type::storage_type;

        storage_type output;
        auto output_it = std::back_inserter(output);
        encode(t, s, output_it);

        auto container_size = output.size();
        auto container_ptr = std::make_shared<container_type>(std::move(output));

        return tensor({container_size}, container_ptr);
    }

    /// Iteratively decode a sequence of position-encoded tokens.
    ///
    /// The result of decoding is sequentially appended to the specified container. If one
    /// of the tokens is not decoded correctly, an exception is raised. All successfully
    /// decoded tokens before thrown exception are left in the container.
    template <forward_iterator<index_type> ForwardIt, std::output_iterator<string_type> OutputIt>
    static void
    decode(const Tokenizer& t, ForwardIt first, ForwardIt last, OutputIt& output)
    {
        using iterator_wrapper = output_iterator_wrapper<string_type, OutputIt>;
        iterator_wrapper output_it(output);

        for (auto id = first; id != last; ++id) {
            t.decode(*id, output_it);
        }
    }

    template <std::output_iterator<string_type> OutputIt>
    static void
    decode(const Tokenizer& t, index_type id, OutputIt& output)
    {
        decode(t, &id, &id + 1, output);
    }

    /// Iteratively decode a sequence of position-encoded tokens.
    ///
    /// All decoded tokens will be concatenated into a resulting string.
    template <forward_iterator<index_type> ForwardIt>
    static string_type
    decode(const Tokenizer& t, ForwardIt first, ForwardIt last)
    {
        std::basic_stringstream<char_type> output;
        std::ostream_iterator<string_type, char_type> output_it(output);

        decode(t, first, last, output_it);
        return output.str();
    }

    static string_type
    decode(const Tokenizer& t, index_type id)
    {
        return decode(t, &id, &id + 1);
    }
};


template <typename Tokenizer>
class tokenizer_wrapper
: public basic_tokenizer<typename Tokenizer::index_type, typename Tokenizer::char_type> {
private:
public:
    using char_type = Tokenizer::char_type;
    using index_type = Tokenizer::index_type;
    using string_type = Tokenizer::string_type;
    using encoding_iterator = Tokenizer::encoding_iterator;
    using decoding_iterator = Tokenizer::decoding_iterator;

    tokenizer_wrapper(Tokenizer&& t)
    : _M_tokenizer(std::move(t))
    {}

    tokenizer_wrapper(const Tokenizer& t)
    : _M_tokenizer(t)
    {}

    void
    encode(tokenkind kind, encoding_iterator& output) const
    {
        return _M_tokenizer.encode(kind, output);
    }

    void
    encode(const string_type& s, encoding_iterator& output) const
    {
        return _M_tokenizer.encode(s, output);
    }

    void
    decode(index_type id, decoding_iterator& output) const
    {
        _M_tokenizer.decode(id, output);
    }


private:
    Tokenizer _M_tokenizer;
};


} // namespace text
} // namespace metalchat
