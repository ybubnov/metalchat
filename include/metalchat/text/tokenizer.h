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


template <typename I, typename T>
concept forward_iterator = std::forward_iterator<I> && std::same_as<std::iter_value_t<I>, T>;


template <typename Tokenizer> struct tokenizer_traits {
    using char_type = Tokenizer::string_type::value_type;
    using index_type = Tokenizer::index_type;
    using string_type = Tokenizer::string_type;
    using string_view_type = std::basic_string_view<char_type>;

    template <std::output_iterator<index_type> OutputIt>
    static OutputIt
    encode(const Tokenizer& t, const string_type& s, OutputIt output)
    {
        return t.encode(s, output);
    }

    template <std::output_iterator<index_type> OutputIt>
    static OutputIt
    encode(const Tokenizer& t, const string_view_type& sv, OutputIt output)
    {
        return t.encode(string_type(sv), output);
    }

    template <std::output_iterator<index_type> OutputIt>
    static OutputIt
    encode(const Tokenizer& t, const char_type* s, OutputIt output)
    {
        return t.encode(string_type(s), output);
    }

    static auto
    encode(const Tokenizer& t, const string_type& s)
    {
        using container_type = vector_memory_container<index_type>;
        using storage_type = container_type::storage_type;

        storage_type output;
        encode(t, s, std::back_inserter(output));

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
    static OutputIt
    decode(const Tokenizer& t, ForwardIt first, ForwardIt last, OutputIt output)
    {
        for (auto id = first; id != last; ++id) {
            output = t.decode(*id, output);
        }
        return output;
    }

    template <std::output_iterator<string_type> OutputIt>
    static OutputIt
    decode(const Tokenizer& t, index_type id, OutputIt output)
    {
        return decode(t, &id, &id + 1, output);
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


} // namespace text
} // namespace metalchat
