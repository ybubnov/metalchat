// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <string>
#include <unordered_map>
#include <utility>


namespace metalchat {
namespace text {


/// Implements GPT-2 style byte-level encoding/decoding for tokenizer preprocessing.
///
/// This codec handles the byte-level BPE (Byte Pair Encoding) character mapping used by GPT-2,
/// LLaMA, and other transformer models. It maps problematic bytes (control characters, spaces,
/// etc.) to displayable Unicode characters in the U+0100-U+01FF range, allowing tokenizer
/// vocabularies to be human-readable while supporting all possible byte values.
///
/// The encoding ensures that:
/// - Printable ASCII characters (33-126) and some extended ASCII remain unchanged.
/// - Control characters and spaces are shifted to higher Unicode code points (>= 256).
/// - Every byte (0-255) has a unique, reversible character representation.
///
/// Example usage:
/// ```cpp
/// using namespace metalchat;
/// text::gpt2_codec codec;
///
/// auto encoding = codec.encode("\tHello World");
/// std::cout << encoding << std::endl;
/// // output: "ĉHelloĠWorld"
///
/// auto decoding = codec.decode(encoded);
/// std::cout << decoding << std::endl;
/// // output: "\tHello World"
/// ```
class gpt2_codec {
public:
    /// The \ref gpt2_codec default constructor.
    gpt2_codec();

    /// Encodes a UTF-8 string by mapping each byte to its corresponding byte-level BPE
    /// character representation.
    ///
    /// \param input The UTF-8 string to encode.
    std::string
    encode(const std::string& input) const;

    /// Decodes a byte-level BPE encoded string back to its original UTF-8 form by reversing
    /// the character-to-byte mapping.
    ///
    /// \param input The encoded string to decode.
    std::string
    decode(const std::string& input) const;

private:
    std::unordered_map<char8_t, char16_t> _M_encoding;
    std::unordered_map<char16_t, char8_t> _M_decoding;
};


} // namespace text
} // namespace metalchat
