// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
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
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <metalchat/container.h>
#include <metalchat/tensor.h>
#include <metalchat/text/regexp.h>


namespace metalchat {
namespace text {


/// Specifies kind of the token.
///
/// Tokens are used to transform a natural language sentences into a vector of integers
/// mapping them to a embedding space of the respective language model. There are specific
/// kinds of tokens that allow to instruct the model for a specific behaviour.
using tokenkind = int32_t;


struct token {
    static constexpr tokenkind regular = 1 << 0;
    static constexpr tokenkind begin_text = 1 << 1;
    static constexpr tokenkind end_text = 1 << 2;
    static constexpr tokenkind reserved = 1 << 3;
    static constexpr tokenkind finetune_right_pad = 1 << 4;
    static constexpr tokenkind begin_header = 1 << 5;
    static constexpr tokenkind end_header = 1 << 6;
    static constexpr tokenkind end_message = 1 << 7;
    static constexpr tokenkind end_turn = 1 << 8;
    static constexpr tokenkind ipython = 1 << 9;
};


/// Returns a reserved token string representation for the specified index.
///
/// \param index An index of the token.
std::string
make_reserved_token(int32_t index);


/// A concept that requires an iterator to dereference a tuple comprised of three elements:
/// (i) token string representation, (ii) token index, and (iii) type of the token.
template <typename It, typename T = std::iterator_traits<It>::value_type>
concept input_token_iterator_t = requires {
    requires std::input_iterator<It>;
    requires std::same_as<T, std::tuple<std::string, int32_t, tokenkind>>;
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
/// ```c++
/// using namespace metalchat;
///
/// byte_pair_encoder tokenizer("tokenizer.model");
/// auto tokens = tokenizer.encode("This is a test sentence.");
/// auto string = tokenizer.decode(tokens.begin(), tokens.end());
///
/// std::cout << string << std::endl;
/// // output: This is a test sentence.
/// ```
template <typename RegularExpression> class byte_pair_encoder {
public:
    using string_type = std::string;

    /// Type used to indicate position of the token in the model (token dictionary).
    using index_type = int32_t;

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
    void
    _M_encode_byte_pairs(const std::string& s, OutputIt output) const
    {
        std::size_t priority_limit = std::numeric_limits<index_type>::max();

        using pair_type = std::pair<index_type, std::size_t>;
        using container_type = std::vector<pair_type>;

        std::priority_queue<pair_type, container_type, std::greater<pair_type>> ordering;
        std::vector<token_segment> encoding;

        // Get the priority from the map, when the key is not presented, return a
        // limit of the priority type.
        auto get_priority = [&](const std::string& key) -> index_type {
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
                *output++ = e.priority;
            }
        }
    }

public:
    /// The \ref byte_pair_encoder copy constructor.
    byte_pair_encoder(const byte_pair_encoder&) = default;

    /// Create an instance of a byte-pair encoder using a base64-encoded token map.
    ///
    /// This constructor reads token map from the specified input stream line-by-line and
    /// decodes base64-decoded tokens.
    ///
    /// \param is An input stream containing tokenizer model.
    /// \param token_regex A regular expression to split the input string into tokens.
    byte_pair_encoder(std::istream& is, const std::string& token_regex)
    : _M_forward_mapping(),
      _M_inverse_mapping(),
      _M_control_mapping(),
      _M_re(std::make_shared<RegularExpression>(token_regex))
    {
        std::string line;
        while (std::getline(is, line)) {
            auto delim = line.find(" ");
            auto key_part = line.substr(0, delim);
            auto value_part = line.substr(delim + 1);

            index_type key = std::stoi(value_part);
            string_type value = base64::decode(key_part);

            insert(value, key);
        }
    }

    template <input_token_iterator_t InputIt>
    byte_pair_encoder(InputIt first, InputIt last, const std::string& token_regex)
    : _M_forward_mapping(),
      _M_inverse_mapping(),
      _M_control_mapping(),
      _M_re(std::make_shared<RegularExpression>(token_regex))
    {
        for (auto it = first; it != last; ++it) {
            auto [key, value, kind] = *it;
            insert(value, key, kind);
        }
    }

    byte_pair_encoder(std::istream&& is, const std::string& token_regex)
    : byte_pair_encoder(is, token_regex)
    {}

    /// Create an instance of byte-pair encoder using a base64-encoded token map.
    ///
    /// This constructor allows to specify a custom token regular expression that fits
    /// best to the target language model.
    ///
    /// \param p A path to the tokenizer model.
    /// \param token_regex A regular expression to split the input string into tokens.
    byte_pair_encoder(const std::filesystem::path& p, const std::string& token_regex)
    : byte_pair_encoder(std::ifstream(p, std::ios::binary), token_regex)
    {}

    /// Convenience constructor, interprets `path` argument as path to the tokenizer model.
    ///
    /// \param path A path to the tokenizer model.
    byte_pair_encoder(const char* path, const std::string& token_regex)
    : byte_pair_encoder(std::filesystem::path(path), token_regex)
    {}

    void
    insert(const std::string& value, index_type key, tokenkind kind = token::regular)
    {
        _M_forward_mapping.insert(std::make_pair(value, key));
        _M_inverse_mapping.insert(std::make_pair(key, value));

        if (kind != token::regular) {
            _M_control_mapping.insert(std::make_pair(kind, key));
        }
    }

    void
    insert_back(const std::string& value, tokenkind kind = token::regular)
    {
        auto key = static_cast<index_type>(size());
        insert(value, key, kind);
    }

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
    void
    encode(const std::string& s, OutputIt output) const
    {
        for (auto match = _M_re->begin(s); match != _M_re->end(); ++match) {
            auto key = (*match);
            if (auto it = _M_forward_mapping.find(key); it != _M_forward_mapping.end()) {
                *output++ = it->second;
            } else {
                _M_encode_byte_pairs(key, output);
            }
        }
    }

    /// Encode a special token.
    ///
    /// Method returns a position of a special token within a tokenizer model.
    index_type
    encode(const tokenkind& kind) const
    {
        if (auto it = _M_control_mapping.find(kind); it != _M_control_mapping.end()) {
            return it->second;
        }
        throw std::invalid_argument(
            std::format("byte_pair_encoder: unknown control token '{}'", kind)
        );
    }

    /// Encode a special token.
    ///
    /// Method encodes the provided special token and pushes the result to the output iterator.
    template <std::output_iterator<index_type> OutputIt>
    void
    encode(const tokenkind& kind, OutputIt output) const
    {
        *output++ = encode(kind);
    }

    auto
    encode(const std::string& s) const
    {
        std::vector<index_type> output;
        encode(s, std::back_inserter(output));
        return to_tensor<index_type>({output.size()}, output.cbegin(), output.cend());
    }

    /// Decode a single position-encoded token to the string representation.
    ///
    /// Method at first attempts to find a token within a model token map, then tries to
    /// query special tokens. In token is not found, method raises an exception.
    const std::string
    decode(index_type id) const
    {
        if (auto tok = _M_inverse_mapping.find(id); tok != _M_inverse_mapping.end()) {
            return tok->second;
        }
        throw std::runtime_error(std::format("byte_pair_encoder: unable to decode id '{}'", id));
    }

    /// Iteratively decode a sequence of position-encoded tokens.
    ///
    /// The result of decoding is sequentially appended to the specified container. If one
    /// of the tokens is not decoded correctly, an exception is raised. All successfully
    /// decoded tokens before thrown exception are left in the container.
    template <std::forward_iterator ForwardIt, std::output_iterator<std::string> OutputIt>
    void
    decode(ForwardIt first, ForwardIt last, OutputIt output) const
    {
        for (auto id = first; id != last; ++id) {
            *output++ = decode(*id);
        }
    }

    /// Iteratively decode a sequence of position-encoded tokens.
    ///
    /// All decoded tokens will be concatenated into a resulting string.
    template <std::forward_iterator ForwardIt>
    std::string
    decode(ForwardIt first, ForwardIt last) const
    {
        std::stringstream output;
        decode(first, last, std::ostream_iterator<std::string>(output));
        return output.str();
    }
};


using bpe = byte_pair_encoder<regexp>;


} // namespace text
} // namespace metalchat
