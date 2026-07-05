// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <iterator>

#include <metalchat/text/unicode.h>


namespace metalchat {
namespace text {


template <typename Tokenizer> class unicode_tokenizer_adaptor : public Tokenizer {
public:
    using char_type = char;
    using index_type = Tokenizer::index_type;
    using string_type = std::string;

    using Tokenizer::Tokenizer;
    using UnicodeCodec = utf8_codec<typename Tokenizer::char_type>;

    /// The \ref unicode_tokenizer_adaptor copy constructor.
    unicode_tokenizer_adaptor(const unicode_tokenizer_adaptor&) = default;

    void
    insert(const string_type& value, index_type id)
    {
        Tokenizer::insert(UnicodeCodec::decode(value), id);
    }

    void
    insert_back(const string_type& value)
    {
        Tokenizer::insert(UnicodeCodec::decode(value));
    }

    template <std::output_iterator<index_type> OutputIt>
    OutputIt
    encode(const string_type& s, OutputIt output) const
    {
        return Tokenizer::encode(UnicodeCodec::decode(s), output);
    }

    template <std::output_iterator<string_type> OutputIt>
    OutputIt
    decode(index_type id, OutputIt output) const
    {
        typename Tokenizer::string_type s;
        Tokenizer::decode(id, &s);

        *output = UnicodeCodec::encode(s);
        ++output;
        return output;
    }
};


} // namespace text
} // namespace metalchat
