#pragma once
#define PCRE2_CODE_UNIT_WIDTH 8

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

#include <cppcodec/base64_rfc4648.hpp>
#include <pcre2.h>

#include <metalchat/container.h>
#include <metalchat/tensor.h>


namespace metalchat {


class re3_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using iterator = re3_iterator;
    using value_type = std::string;
    using reference = value_type&;
    using pointer = value_type*;
    using difference_type = std::ptrdiff_t;

    re3_iterator();

    re3_iterator(const std::shared_ptr<pcre2_code> re, const std::string& input);

    ~re3_iterator();

    iterator&
    operator++();

    value_type
    operator*();

    bool
    operator!=(const iterator& rhs);

private:
    const std::shared_ptr<pcre2_code> _m_re = nullptr;
    pcre2_match_data* _m_data = nullptr;
    const PCRE2_SPTR _m_subject = nullptr;
    const PCRE2_SIZE _m_subject_length = 0;
    PCRE2_SIZE _m_offset = 0;
    bool _m_end = false;

    value_type
    get();

    void
    next();
};


class re3 {
private:
    std::shared_ptr<pcre2_code> _m_re = nullptr;

    static constexpr std::size_t error_buffer_size = 256;

public:
    re3(const std::string& regex);

    re3_iterator
    begin(const std::string& input) const;

    re3_iterator
    end() const;
};


template <typename T>
concept push_back_container = requires(T t) {
    typename T::value_type;
    { t.push_back(typename T::value_type{}) } -> std::same_as<void>;
};


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


class byte_pair_encoder {
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
    using base64 = cppcodec::base64_rfc4648;

    std::unordered_map<string_type, index_type> _m_fmap;
    std::unordered_map<index_type, string_type> _m_rmap;

    re3 _m_re;

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
    template <push_back_container PushBackContainer>
    void
    _m_encode_byte_pairs(const std::string& s, PushBackContainer& ids) const
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
            ids.push_back(_m_fmap.at(key));
        }
    }

public:
    static constexpr index_type pad = -1;

    /// Number of special tokens used to prepare the input for the model.
    static constexpr const index_type nspecial = 256;

    /// Create an instance of a byte-pair encoder using a base64-encoded token map.
    ///
    /// Such map is distributed altogether with, for example, Llama model and is called
    /// `tokenizer.model`. When the provided file does not exist or has invalid format,
    /// constructor will raise an exception.
    byte_pair_encoder(const std::filesystem::path& p);

    /// Encode the provided string into tokens.
    ///
    /// This method iteratively splits the string into tokens and then appends a corresponding
    /// token index into end of the provided container `ids`. When the token is not presented
    /// in the token dictionary, it is divided into byte-pairs, then index of the byte pair is
    /// appended to the end of the container.
    template <push_back_container PushBackContainer>
    void
    encode(const std::string& s, PushBackContainer& ids) const
    {
        for (auto match = _m_re.begin(s); match != _m_re.end(); ++match) {
            auto key = (*match);
            if (auto it = _m_fmap.find(key); it != _m_fmap.end()) {
                ids.push_back(it->second);
            } else {
                _m_encode_byte_pairs(key, ids);
            }
        }
    }

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

    template <push_back_container PushBackContainer>
    void
    encode(const special_token& s, PushBackContainer& ids) const
    {
        ids.push_back(encode(s));
    }

    auto
    encode(const std::string& s) const
    {
        std::vector<index_type> ids;
        encode(s, ids);
        return to_tensor<index_type>({ids.size()}, ids.cbegin(), ids.cend());
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
    template <std::forward_iterator ForwardIt, push_back_container PushBackContainer>
    void
    decode(ForwardIt first, ForwardIt last, PushBackContainer output) const
    {
        for (auto id = first; id != last; ++id) {
            output.push_back(decode(*id));
        }
    }

    template <std::forward_iterator ForwardIt>
    std::string
    decode(ForwardIt first, ForwardIt last) const
    {
        std::stringstream output;

        struct _StringStreamContainer {
            void
            push_back(const string_type& s)
            {
                (*_m_ss) << s;
            }

            using value_type = string_type;
            std::stringstream* _m_ss;
        } __stream{&output};

        decode(first, last, __stream);
        return output.str();
    }
};


} // namespace metalchat
