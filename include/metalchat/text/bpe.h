// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <metalchat/text/regexp.h>
#include <metalchat/text/tokenizer.h>


namespace metalchat {
namespace text {


/// Returns a reserved token string representation for the specified index.
///
/// \param index An index of the token.
std::string
make_reserved_token(int32_t index);


/// A concept that requires an iterator to dereference a tuple comprised of three elements:
/// (i) token string representation, (ii) token index, and (iii) type of the token.
template <typename It, typename CharT, typename T = std::iterator_traits<It>::value_type>
concept input_token_iterator_t = requires {
    requires std::input_iterator<It>;
    requires std::same_as<T, std::tuple<std::basic_string<CharT>, int32_t, tokenkind>>;
};


/// Token encoder that splits arbitrary utf-8 encoded string into a sequence of tokens
/// that could be used to run the inference of a language transformer. The approach and the
/// implementation is inspired by [tiktoken](https://github.com/openai/tiktoken).
///
/// Constructors of this class require a path to a token map, such map is distributed altogether
/// with, for example, Llama model and is called `tokenizer.model`. When the provided file does
/// not exist or has invalid format, constructor will raise an exception.
///
/// Here is an example of a tokenizer model: in the first column - a base64-encoded token, in the
/// second column - a token identifier of `byte_pair_encoder::index_type`):
///
/// ```txt
/// 4LmM4LiB4Lij 0
/// zrbOsQ== 1
/// IOuNlOyasQ== 2
/// 2YjZhNin2Ko= 3
/// ```
///
/// Consider the following basic example:
/// ```cpp
/// using namespace metalchat::text;
///
/// using Tokenizer = byte_pair_encoder<char>;
/// using TokenizerTraits = tokenizer_traits<Tokenizer>;
///
/// Tokenizer tokenizer("tokenizer.model");
/// auto tokens = TokenizerTraits::encode(tokenizer, "This is a test sentence.");
/// auto string = TokenizerTraits::decode(tokenizer, tokens.begin(), tokens.end());
///
/// std::cout << string << std::endl;
/// // output: This is a test sentence.
/// ```
template <typename CharT, typename RegularExpression = unicode_regexp<CharT>>
class byte_pair_encoder {
public:
    /// Type used to indicate position of the token in the model (token dictionary).
    using char_type = CharT;
    using index_type = int32_t;
    using string_type = std::basic_string<CharT>;

private:
    std::unordered_map<string_type, index_type, _StringHash> _M_forward_mapping;
    std::unordered_map<index_type, string_type> _M_inverse_mapping;
    std::unordered_map<tokenkind, index_type> _M_control_mapping;

    std::shared_ptr<RegularExpression> _M_re;

    /// This structure is used in the byte-pair merging algorithm.
    struct token_segment {
        index_type priority;
        std::size_t end;

        token_segment(index_type p, std::size_t e)
        : priority(p),
          end(e)
        {}
    };

    /// Encode the specified string by joining byte pairs.
    ///
    /// The algorithm works like following:
    /// 1. Compute for every byte pair encoding (an index from the token map).
    /// 2. Then iterate through those byte pair encodings in the order from lowest
    ///    priority to the highest. Where priority is an index in the token map.
    /// 3. Join to adjacent encodings, only when such encoding exists.
    /// 4. Then push encodings to the specified container of identifiers.
    template <std::output_iterator<index_type> OutputIt>
    OutputIt
    _M_encode_unicode_pairs(const string_type& s, OutputIt output) const
    {
        std::size_t priority_limit = std::numeric_limits<index_type>::max();

        using pair_type = std::pair<index_type, std::size_t>;
        using container_type = std::vector<pair_type>;

        std::priority_queue<pair_type, container_type, std::greater<pair_type>> ordering;
        std::vector<token_segment> encoding;

        // Get the priority from the map, when the key is not presented, return a
        // limit of the priority type.
        auto get_priority = [&](const string_type& key) -> index_type {
            if (auto it = _M_forward_mapping.find(key); it != _M_forward_mapping.end()) {
                return it->second;
            }
            return priority_limit;
        };

        for (std::size_t i = 0; i < s.size() - 1; i++) {
            auto key = s.substr(i, 1);
            auto priority = get_priority(key);

            ordering.emplace(priority, i);
            encoding.emplace_back(priority, i + 1);
        }

        encoding.emplace_back(priority_limit, s.size());

        while (!ordering.empty()) {
            auto [priority, begin] = ordering.top();
            auto next = encoding[begin].end;
            ordering.pop();

            if (encoding[begin].priority >= priority_limit || next >= encoding.size()) {
                continue;
            }

            auto end = encoding[next].end;
            auto key = s.substr(begin, end - begin);

            auto merged_priority = get_priority(key);
            if (merged_priority >= priority_limit) {
                continue;
            }

            // Merge elements, then push a merge into the queue for further processing.
            ordering.emplace(merged_priority, begin);

            encoding[begin].priority = merged_priority;
            encoding[begin].end = end;
            encoding[next].priority = priority_limit;
        }

        for (auto& e : encoding) {
            if (e.priority < priority_limit) {
                *output = e.priority;
                ++output;
            }
        }

        return output;
    }

public:
    /// The \ref byte_pair_encoder copy constructor.
    byte_pair_encoder(const byte_pair_encoder&) = default;

    byte_pair_encoder(const string_type& token_regex)
        requires std::constructible_from<RegularExpression, string_type>
    : _M_forward_mapping(),
      _M_inverse_mapping(),
      _M_control_mapping(),
      _M_re(std::make_shared<RegularExpression>(token_regex))
    {}

    /// Create an instance of a byte-pair encoder using a base64-encoded token map.
    ///
    /// This constructor reads token map from the specified input stream line-by-line and
    /// decodes base64-decoded tokens.
    ///
    /// \param is An input stream containing tokenizer model.
    /// \param token_regex A regular expression to split the input string into tokens.
    byte_pair_encoder(std::basic_istream<CharT>& is, const string_type& token_regex)
    : byte_pair_encoder(token_regex)
    {
        string_type line;
        while (std::getline(is, line)) {
            auto delim = line.find(' ');
            auto key_part = line.substr(0, delim);
            auto value_part = line.substr(delim + 1);

            index_type key = std::stoi(value_part);
            string_type value = base64::decode(key_part);

            insert(value, key);
        }
    }

    template <input_token_iterator_t<CharT> InputIt>
    byte_pair_encoder(InputIt first, InputIt last, const string_type& token_regex)
    : byte_pair_encoder(token_regex)
    {
        for (auto it = first; it != last; ++it) {
            auto [key, value, kind] = *it;
            insert(value, key, kind);
        }
    }

    byte_pair_encoder(std::istream&& is, const string_type& token_regex)
    : byte_pair_encoder(is, token_regex)
    {}

    /// Create an instance of byte-pair encoder using a base64-encoded token map.
    ///
    /// This constructor allows to specify a custom token regular expression that fits
    /// best to the target language model.
    ///
    /// \param p A path to the tokenizer model.
    /// \param token_regex A regular expression to split the input string into tokens.
    byte_pair_encoder(const std::filesystem::path& p, const string_type& token_regex)
    : byte_pair_encoder(std::ifstream(p, std::ios::binary), token_regex)
    {}

    /// Convenience constructor, interprets `path` argument as path to the tokenizer model.
    ///
    /// \param path A path to the tokenizer model.
    /// \param token_regex A regular expression to split the input string into tokens.
    byte_pair_encoder(const char* path, const string_type& token_regex)
    : byte_pair_encoder(std::filesystem::path(path), token_regex)
    {}

    /// Insert a new token-pair into the encoder.
    ///
    /// \param value A string representation of a token.
    /// \param key Target encoding of a token (a position in the token embedding).
    /// \param kind A type of the token, used for special token binding.
    void
    insert(const string_type& value, index_type key, tokenkind kind = token::regular)
    {
        _M_forward_mapping.insert_or_assign(value, key);
        _M_inverse_mapping.insert_or_assign(key, value);

        if (kind != token::regular) {
            _M_control_mapping.insert_or_assign(kind, key);
        }
    }

    /// Insert a new token by binding it to the last position (in the token embedding).
    ///
    /// \param value A string representation of a token.
    /// \param kind A type of the token, used for special token binding.
    void
    insert_back(const string_type& value, tokenkind kind = token::regular)
    {
        auto key = static_cast<index_type>(size());
        insert(value, key, kind);
    }

    /// Returns the number of all available tokens in the encoder.
    std::size_t
    size() const
    {
        return _M_forward_mapping.size();
    }

    /// Encode the provided string into tokens.
    ///
    /// This method iteratively splits the string into tokens and then appends a corresponding
    /// token index into end of the provided iterator `output`. When the token is not presented
    /// in the token dictionary, it is divided into byte-pairs, then index of the byte pair is
    /// appended to the end of the container.
    template <std::output_iterator<index_type> OutputIt>
    OutputIt
    encode(const string_type& s, OutputIt output) const
    {
        for (auto match = _M_re->begin(s); match != _M_re->end(); ++match) {
            auto key = (*match);
            if (auto it = _M_forward_mapping.find(key); it != _M_forward_mapping.end()) {
                *output = it->second;
                ++output;
            } else {
                output = _M_encode_unicode_pairs(key, output);
            }
        }
        return output;
    }

    /// Encode a special token.
    ///
    /// Method returns a position of a special token within a tokenizer model. When a token is
    /// a `token::regular` kind, then method raises an exception. Regular token encoding is
    /// available through \ref encode(const string_type&, OutputIt) const method.
    template <std::output_iterator<index_type> OutputIt>
    OutputIt
    encode(tokenkind kind, OutputIt output) const
    {
        if (auto it = _M_control_mapping.find(kind); it != _M_control_mapping.end()) {
            *output = it->second;
            ++output;
            return output;
        }
        throw std::invalid_argument(
            std::format("byte_pair_encoder: unknown control token '{}'", kind)
        );
    }

    /// Decode a single position-encoded token to the string representation.
    ///
    /// Method at first attempts to find a token within a model token map, then tries to
    /// query special tokens. In token is not found, method raises an exception.
    template <std::output_iterator<string_type> OutputIt>
    OutputIt
    decode(index_type id, OutputIt output) const
    {
        if (auto tok = _M_inverse_mapping.find(id); tok != _M_inverse_mapping.end()) {
            *output = tok->second;
            ++output;
            return output;
        }
        throw std::runtime_error(std::format("byte_pair_encoder: unable to decode id '{}'", id));
    }
};


using bpe = byte_pair_encoder<char>;


} // namespace text
} // namespace metalchat
