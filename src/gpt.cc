// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <algorithm>
#include <codecvt>
#include <iostream>
#include <locale>
#include <queue>
#include <sstream>

#include <metalchat/text/gpt.h>


namespace metalchat {
namespace text {


gpt2_codec::gpt2_codec()
: _M_encoding(),
  _M_decoding()
{
    using range_type = std::pair<std::size_t, std::size_t>;

    std::queue<range_type> ranges({{0x21, 0x7e}, {0xa1, 0xac}, {0xae, 0xff}});
    std::size_t offset = 0;

    constexpr std::size_t cardinality = 1 << std::numeric_limits<char8_t>::digits;

    for (std::size_t i = 0; i < cardinality; i++) {
        auto& range = ranges.front();

        if (i >= range.first && i <= range.second) {
            _M_encoding.insert_or_assign(char8_t(i), char16_t(i));
            _M_decoding.insert_or_assign(char16_t(i), char8_t(i));
        } else {
            auto encoding = offset + cardinality;
            offset++;

            _M_encoding.insert_or_assign(char8_t(i), char16_t(encoding));
            _M_decoding.insert_or_assign(char16_t(encoding), char8_t(i));
        }

        if (i == range.second) {
            ranges.pop();
        }
    }
}


std::string
gpt2_codec::encode(const std::string& input) const
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> converter;
#pragma clang diagnostic pop

    std::basic_stringstream<char16_t> output;

    const auto bytes = reinterpret_cast<const char8_t*>(input.data());
    const auto bytes_size = input.size() * sizeof(char);

    for (std::size_t i = 0; i < bytes_size; i++) {
        auto byte = bytes[i];
        if (auto it = _M_encoding.find(byte); it != _M_encoding.end()) {
            output.put(it->second);
        } else {
            output.put(char16_t(byte));
        }
    }

    return converter.to_bytes(output.str());
}


std::string
gpt2_codec::decode(const std::string& input) const
{
    std::basic_stringstream<char> output;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> converter;
#pragma clang diagnostic pop

    auto winput = converter.from_bytes(input);

    const auto runes = winput.data();
    const auto runes_size = winput.size();

    for (std::size_t i = 0; i < runes_size; i++) {
        auto rune = runes[i];
        if (auto it = _M_decoding.find(rune); it != _M_decoding.end()) {
            output.put(char(it->second));
        } else {
            output.put(char(rune));
        }
    }

    return output.str();
}


} // namespace text
} // namespace metalchat
