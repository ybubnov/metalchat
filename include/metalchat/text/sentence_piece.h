// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <algorithm>

#include <metalchat/text/bpe.h>


namespace metalchat {
namespace text {


/// A tokenizer that applies byte-pair tokenizer directly on unicode text.
class sentence_piece {
public:
    using char_type = char32_t;
    using Tokenizer = byte_pair_encoder<char_type>;

    using string_type = Tokenizer::string_type;
    using index_type = Tokenizer::index_type;

    /// The \ref sentence_piece copy constructor.
    sentence_piece(const sentence_piece&) = default;

    /// The \ref sentence_piece default constructor.
    sentence_piece()
    : _M_bpe(UR"(.*)")
    {}

    template <input_token_iterator_t<char_type> InputIt>
    sentence_piece(InputIt first, InputIt last)
    : _M_bpe(first, last, UR"(.*)")
    {}

    /// \copydoc byte_pair_encoder::insert
    void
    insert(const string_type& value, index_type key, tokenkind kind = token::regular)
    {
        _M_bpe.insert(value, key, kind);
    }

    /// \copydoc byte_pair_encoder::insert_back
    void
    insert_back(const string_type& value, tokenkind kind = token::regular)
    {
        _M_bpe.insert_back(value, kind);
    }

    /// \copydoc byte_pair_encoder::size
    std::size_t
    size() const
    {
        return _M_bpe.size();
    }

    /// Encode the provided string into tokens.
    ///
    /// The method replaces all white space characters with a special unicode symbols, and
    /// then encodes the whole sequence using byte-pair encoding.
    template <std::output_iterator<index_type> OutputIt>
    void
    encode(const string_type& s, OutputIt& output) const
    {
        auto input = s;
        std::replace(input.begin(), input.end(), whitespace_forward, whitespace_inverse);
        _M_bpe.encode(input, output);
    }

    /// \copydoc byte_pair_encoder::encode(tokenkind) const
    index_type
    encode(tokenkind kind) const
    {
        return _M_bpe.encode(kind);
    }

    /// \copydoc byte_pair_encoder::encode(tokenkind, OutputIt) const
    template <std::output_iterator<index_type> OutputIt>
    void
    encode(tokenkind kind, OutputIt& output) const
    {
        _M_bpe.encode(kind, output);
    }

    /// Decode a single position-encoded token to the string representation.
    ///
    /// Method replaces all whitespace-replacement unicode code points with a unicode code
    /// point of the regular white space.
    string_type
    decode(index_type id) const
    {
        auto s = _M_bpe.decode(id);
        std::replace(s.begin(), s.end(), whitespace_inverse, whitespace_forward);
        return s;
    }

    /// \copydoc byte_pair_encoder::decode(ForwardIt, ForwardIt, OutputIt) const
    template <std::forward_iterator ForwardIt, std::output_iterator<string_type> OutputIt>
    void
    decode(ForwardIt first, ForwardIt last, OutputIt& output) const
    {
        for (auto id = first; id != last; ++id) {
            *output++ = decode(*id);
        }
    }

    /// \copydoc byte_pair_encoder::decode(ForwardIt, ForwardIt) const
    template <std::forward_iterator ForwardIt>
    string_type
    decode(ForwardIt first, ForwardIt last) const
    {
        std::basic_stringstream<char_type> output;
        decode(first, last, std::ostream_iterator<string_type, char_type>(output));
        return output.str();
    }

private:
    static constexpr char_type whitespace_forward = U' ';
    static constexpr char_type whitespace_inverse = U'▁';

    Tokenizer _M_bpe;
};


} // namespace text
} // namespace metalchat
