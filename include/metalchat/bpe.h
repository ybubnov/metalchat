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
#include <unordered_map>
#include <vector>

#include <metalchat/container.h>
#include <metalchat/tensor.h>


namespace metalchat {


class regexp_iterator;


class regexp {
public:
    regexp(const std::string& regex);
    regexp(const char* regex);

    regexp_iterator
    begin(const std::string&) const;

    regexp_iterator
    end() const;

private:
    struct _RegularExpression;
    std::shared_ptr<_RegularExpression> _m_impl;

    friend class regexp_iterator;
};


/// Regular expression iterator.
///
/// This iterator is used to provide convenient interface to access match group
/// data. So every match is considered an element of the backing container and
/// this iterator returns matches sequentially until the last match.
class regexp_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::string;
    using reference = value_type&;
    using pointer = value_type*;
    using difference_type = std::ptrdiff_t;

    /// Advance the iterator to the next regular expression match.
    regexp_iterator&
    operator++();

    /// Return the current match of the regular expression.
    ///
    /// The method throws `std::runtime_error` when it is called on a terminated iterator.
    value_type
    operator*();

    /// Compares two regular expression iterators.
    ///
    /// The implementation is naive for simplicity reasons, and only compares
    /// the ends of iterators.
    bool
    operator!=(const regexp_iterator&);

    /// Initialize the end-of-match-group iterator.
    regexp_iterator();

    /// Initializes the iterators, stores the address of `regexp` in data member, and performs
    /// the finds the first match from the input string to initialize match group members.
    regexp_iterator(const regexp& regex, const std::string& input);

private:
    struct _RegularExpressionIterator;
    std::shared_ptr<_RegularExpressionIterator> _m_impl;

    /// Advance the iterator to the next match group.
    void
    next();

    /// Get the current match group value.
    value_type
    get();

    friend class regexp;
};


class base64 {
public:
    static std::string
    decode(const std::string&);
};


/// Special token is an enumeration of token values used to produce complex prompts for a
/// language model.
enum special_token {
    begin_text,
    end_text,
    reserved0,
    reserved1,
    finetune_right_pad,
    reserved2,
    begin_header,
    end_header,
    end_message,
    end_turn,
    ipython,
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
template <typename RegularExpression = regexp> class byte_pair_encoder {
public:
    using string_type = std::string;

    /// Type used to indicate position of the token in the model (token dictionary).
    using index_type = int32_t;

    static std::string
    make_reserved_token(const index_type& token_id)
    {
        return std::format("<|reserved_special_token_{}|>", token_id);
    }

    const std::unordered_map<index_type, std::string> special_tokens = {
        {special_token::begin_text, "<|begin_of_text|>"},
        {special_token::end_text, "<|end_of_text|>"},
        {special_token::reserved0, make_reserved_token(0)},
        {special_token::reserved1, make_reserved_token(1)},
        {special_token::finetune_right_pad, "<|finetune_right_pad_id|>"},
        {special_token::reserved2, make_reserved_token(2)},
        {special_token::begin_header, "<|start_header_id|>"},
        {special_token::end_header, "<|end_header_id|>"},
        {special_token::end_message, "<|eom_id|>"},
        {special_token::end_turn, "<|eot_id|>"},
        {special_token::ipython, "<|python_tag|>"},
    };

private:
    std::unordered_map<string_type, index_type, _StringHash> _m_fmap;
    std::unordered_map<index_type, string_type> _m_rmap;

    std::shared_ptr<RegularExpression> _m_re;

    /// A regular expression string that is used to split the input text into tokens.
    static constexpr const char* token_regex = (R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|)"
                                                R"([^\r\n\p{L}\p{N}]?\p{L}+|)"
                                                R"(\p{N}{1,3}|)"
                                                R"( ?[^\s\p{L}\p{N}]+[\r\n]*|)"
                                                R"(\s*[\r\n]+|)"
                                                R"(\s+(?!\S)|)"
                                                R"(\s+)");

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
    _m_encode_byte_pairs(const std::string& s, OutputIt output) const
    {
        std::size_t priority_limit = std::numeric_limits<index_type>::max();

        using pair_type = std::pair<index_type, std::size_t>;

        std::priority_queue<pair_type, std::vector<pair_type>, std::greater<pair_type>> ordering;
        std::vector<pair_type> encoding;

        // Get the priority from the map, when the key is not resented, return a
        // limit of the priority type.
        auto get_priority = [&](const std::string& key) -> index_type {
            if (auto it = _m_fmap.find(key); it != _m_fmap.end()) {
                return it->second;
            }
            return priority_limit;
        };

        for (std::size_t i = 0; i < s.size() - 1; i++) {
            auto key = s.substr(i, 2);
            auto priority = get_priority(key);

            ordering.emplace(priority, i);
            encoding.emplace_back(priority, i);
        }

        while (!ordering.empty()) {
            auto [priority, i] = ordering.top();
            ordering.pop();
            if (i >= encoding.size() - 1) {
                continue;
            }

            auto begin = encoding[i].second;
            auto end = encoding[i + 1].second;
            auto key = s.substr(begin, end - begin);

            auto merged_priority = get_priority(key);

            encoding[i].first = merged_priority;
            encoding.erase(encoding.begin() + i + 1);

            // Merge elements, then push a merge into the queue for further processing.
            if (merged_priority < priority_limit) {
                ordering.emplace(merged_priority, i);
            }
        }

        for (std::size_t i = 0; i < encoding.size() - 1; i++) {
            auto begin = encoding[i].second;
            auto end = encoding[i + 1].second;
            auto key = s.substr(begin, end - begin);
            *output++ = _m_fmap.at(key);
        }
    }

public:
    static constexpr index_type pad = -1;

    /// Number of special tokens used to prepare the input for the model.
    static constexpr const index_type nspecial = 256;

    byte_pair_encoder(const byte_pair_encoder&) = default;

    /// Create an instance of byte-pair encoder using a base64-encoded token map.
    ///
    /// This constructor allows to specify a custom token regular expression that fits
    /// best to the target language model.
    ///
    /// \param p A path to the tokenizer model.
    /// \param token_regex A regular expression to split the input string into tokens.
    byte_pair_encoder(const std::filesystem::path& p, const std::string& token_regex)
    : _m_fmap(),
      _m_rmap(),
      _m_re(std::make_shared<RegularExpression>(token_regex))
    {
        std::ifstream file(p, std::ios::binary);
        if (!file.is_open()) {
            throw std::invalid_argument(std::format("unable to open file '{}'", p.string()));
        }

        std::string line;
        while (std::getline(file, line)) {
            auto delim = line.find(" ");
            auto key_part = line.substr(0, delim);
            auto value_part = line.substr(delim + 1);

            index_type key = std::stoi(value_part);
            string_type value = base64::decode(key_part);

            _m_fmap.insert(std::make_pair(value, key));
            _m_rmap.insert(std::make_pair(key, value));
        }
    }


    /// Create an instance of a byte-pair encoder using a base64-encoded token map.
    ///
    /// By default, encoder uses regular expression for Meta Llama 3.2 model.
    ///
    /// \param p A path to the tokenizer model.
    byte_pair_encoder(const std::filesystem::path& p)
    : byte_pair_encoder(p, byte_pair_encoder::token_regex)
    {}

    /// Convenience constructor, interprets `path` argument as path to the tokenizer model.
    ///
    /// \param path A path to the tokenizer model.
    byte_pair_encoder(const char* path)
    : byte_pair_encoder(std::filesystem::path(path))
    {}

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
        for (auto match = _m_re->begin(s); match != _m_re->end(); ++match) {
            auto key = (*match);
            if (auto it = _m_fmap.find(key); it != _m_fmap.end()) {
                *output++ = it->second;
            } else {
                _m_encode_byte_pairs(key, output);
            }
        }
    }

    /// Encode a special token.
    ///
    /// Method returns a position of a special token within a tokenizer model.
    index_type
    encode(const special_token& s) const
    {
        auto index = static_cast<index_type>(s);
        if (index > nspecial) {
            throw std::invalid_argument(
                std::format("byte_pair_encoder: unknown special token '{}'", index)
            );
        }
        return _m_fmap.size() + index;
    }

    /// Encode a special token.
    ///
    /// Method encodes the provided special token and pushes the result to the output iterator.
    template <std::output_iterator<index_type> OutputIt>
    void
    encode(const special_token& s, OutputIt output) const
    {
        *output++ = encode(s);
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
        if (auto tok = _m_rmap.find(id); tok != _m_rmap.end()) {
            return tok->second;
        }
        if (auto tok = special_tokens.find(id - _m_rmap.size()); tok != special_tokens.end()) {
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


} // namespace metalchat
