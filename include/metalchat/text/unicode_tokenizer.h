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

    using Tokenizer::Tokenizer;

    /// The \ref unicode_tokenizer_adaptor copy constructor.
    unicode_tokenizer_adaptor(const unicode_tokenizer_adaptor&) = default;

    template <std::output_iterator<index_type> OutputIt>
    OutputIt
    encode(const string_type& s, OutputIt output) const
    {
        return Tokenizer::encode(decode_bytes(s), output);
    }

    template <std::output_iterator<index_type> OutputIt>
    OutputIt
    encode(tokenkind kind, OutputIt output) const
    {
        return Tokenizer::encode(kind, output);
    }

    template <std::output_iterator<string_type> OutputIt>
    OutputIt
    decode(index_type id, decoding_iterator& output) const
    {
        value_type s;
        Tokenizer::decode(id, &s);

        *output = encode_bytes(s);
        ++output;
        return output;
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
