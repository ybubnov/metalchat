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
    using char_type = Tokenizer::char_type;
    using index_type = Tokenizer::index_type;

    using Tokenizer::decode;
    using Tokenizer::encode;
    using Tokenizer::Tokenizer;

    /// The \ref unicode_tokenizer_adaptor copy constructor.
    unicode_tokenizer_adaptor(const unicode_tokenizer_adaptor&) = default;

    template <std::output_iterator<index_type> OutputIt>
    void
    encode(const std::string& s, OutputIt output) const
    {
        encode(decode_bytes<char_type>(s), output);
    }

    std::string
    decode(index_type id) const
    {
        return encode_bytes(Tokenizer::decode(id));
    }

    template <std::forward_iterator ForwardIt, std::output_iterator<std::string> OutputIt>
    void
    decode(ForwardIt first, ForwardIt last, OutputIt output) const
    {
        for (auto id = first; id != last; ++id) {
            *output++ = decode(*id);
        }
    }

    template <std::forward_iterator ForwardIt>
    std::string
    decode(ForwardIt first, ForwardIt last) const
    {
        return encode_bytes(Tokenizer::decode(first, last));
    }

private:
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    using codecvt_type = std::codecvt_utf8<char_type>;
    using convert_type = std::wstring_convert<codecvt_type, char_type>;
#pragma clang diagnostic pop

    template <typename CharT>
    static std::string
    encode_bytes(const std::basic_string<CharT>& s)
    {
        convert_type convert;
        return convert.to_bytes(s);
    }

    template <typename CharT>
    static std::basic_string<CharT>
    decode_bytes(const std::string& b)
    {
        convert_type convert;
        return convert.from_bytes(b);
    }
};


} // namespace text
} // namespace metalchat
