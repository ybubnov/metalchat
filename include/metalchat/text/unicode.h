// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <string>


namespace metalchat {
namespace text {


class base64_codec {
public:
    static std::string
    decode(const std::string&);
};


template <typename CharT> struct utf8_codec {
    static std::basic_string<char>
    encode(const std::basic_string<CharT>& s);

    static std::basic_string<CharT>
    decode(const std::basic_string<char>& b);
};


template <> struct utf8_codec<char16_t> {
    static std::basic_string<char>
    encode(const std::basic_string<char16_t>& s);

    static std::basic_string<char16_t>
    decode(const std::basic_string<char>& b);
};


template <> struct utf8_codec<char32_t> {
    static std::basic_string<char>
    encode(const std::basic_string<char32_t>& s);

    static std::basic_string<char32_t>
    decode(const std::basic_string<char>& b);
};


} // namespace text
} // namespace metalchat
