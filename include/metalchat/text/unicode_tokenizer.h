// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <codecvt>
#include <iterator>
#include <locale>


namespace metalchat {
namespace text {


template <typename Tokenizer> class unicode_tokenizer_adaptor : public Tokenizer {
public:
    using char_type = char;
    using index_type = Tokenizer::index_type;
    using string_type = std::string;

    using encoding_iterator = basic_output_iterator<index_type>;
    using decoding_iterator = basic_output_iterator<string_type>;

    using Tokenizer::Tokenizer;

    /// The \ref unicode_tokenizer_adaptor copy constructor.
    unicode_tokenizer_adaptor(const unicode_tokenizer_adaptor&) = default;

    void
    encode(const string_type& s, encoding_iterator& output) const
    {
        Tokenizer::encode(decode_bytes(s), output);
    }

    void
    encode(tokenkind kind, encoding_iterator& output) const
    {
        Tokenizer::encode(kind, output);
    }

    void
    decode(index_type id, decoding_iterator& output) const
    {
        using value_type = Tokenizer::string_type;
        using iterator = value_type*;
        using iterator_wrapper = output_iterator_wrapper<value_type, iterator>;

        value_type s;
        value_type* s_ptr = &s;
        iterator_wrapper output_it(s_ptr);

        Tokenizer::decode(id, output_it);
        *output = encode_bytes(s);
        ++output;
    }

private:
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    using rune_type = Tokenizer::char_type;
    using codecvt_type = std::codecvt_utf8<rune_type>;
    using convert_type = std::wstring_convert<codecvt_type, rune_type>;
#pragma clang diagnostic pop

    static string_type
    encode_bytes(const std::basic_string<rune_type>& s)
    {
        convert_type convert;
        return convert.to_bytes(s);
    }

    static std::basic_string<rune_type>
    decode_bytes(const string_type& b)
    {
        convert_type convert;
        return convert.from_bytes(b);
    }
};


} // namespace text
} // namespace metalchat
