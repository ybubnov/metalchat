#pragma once

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cppcodec/base64_rfc4648.hpp>
#include <unicode/regex.h>

#include <metalchat/container.h>
#include <metalchat/tensor.h>


namespace metalchat {


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
    python,
};


class bpe {
public:
    using string_type = std::string;
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
        {special_token::python, "<|python_tag|>"},
    };

private:
    using base64 = cppcodec::base64_rfc4648;

    std::unordered_map<string_type, index_type> _m_fmap;
    std::unordered_map<index_type, string_type> _m_rmap;

    std::unique_ptr<icu::RegexMatcher> _m_re;

public:
    static constexpr index_type pad = -1;

    /// A regular expression string that is used to split the input text into tokens.
    static constexpr const char* token_regex = (R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|)"
                                                R"([^\r\n\p{L}\p{N}]?\p{L}+|)"
                                                R"(\p{N}{1,3}|)"
                                                R"( ?[^\s\p{L}\p{N}]+[\r\n]*|)"
                                                R"(\s*[\r\n]+|)"
                                                R"(\s+(?!\S)|)"
                                                R"(\s+)");

    /// Number of special tokens used to prepare the input for the model.
    static constexpr const std::size_t nspecial = 256;

    bpe(const std::filesystem::path& p)
    : _m_fmap(),
      _m_rmap(),
      _m_re(nullptr)
    {
        UErrorCode status = U_ZERO_ERROR;
        _m_re = std::make_unique<icu::RegexMatcher>(token_regex, 0, status);

        if (U_FAILURE(status)) {
            throw std::invalid_argument(
                std::format("unable to compile regexp '{}'", u_errorName(status))
            );
        }

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
            string_type value = base64::decode<string_type>(key_part);

            _m_fmap.insert(std::make_pair(value, key));
            _m_rmap.insert(std::make_pair(key, value));
        }
    }

    template <push_back_container PushBackContainer>
    void
    encode(const std::string& s, PushBackContainer& ids)
    {
        _m_re->reset(icu::UnicodeString::fromUTF8(s));

        UErrorCode status = U_ZERO_ERROR;
        while (_m_re->find(status)) {
            if (U_FAILURE(status)) {
                throw std::runtime_error(
                    std::format("unable to find next token '{}'", u_errorName(status))
                );
            }

            auto match = _m_re->group(status);
            if (U_FAILURE(status)) {
                throw std::runtime_error(
                    std::format("unable to match a token '{}'", u_errorName(status))
                );
            }

            std::string key;
            key = match.toUTF8String(key);

            if (auto it = _m_fmap.find(key); it != _m_fmap.end()) {
                ids.push_back(it->second);
            } else {
                throw std::runtime_error(std::format("bpe: byte-pair merging is not implemented"));
            }
            // TODO: else, concatenate byte pairs.
        }
    }

    template <push_back_container PushBackContainer>
    void
    encode(const special_token& s, PushBackContainer& ids)
    {
        auto index = static_cast<index_type>(s);
        if (index > nspecial) {
            throw std::invalid_argument(std::format("unknown special token '{}'", index));
        }

        ids.push_back(_m_fmap.size() + index);
    }

    auto
    encode(const std::string& s)
    {
        std::vector<index_type> ids;
        encode(s, ids);
        return to_tensor<index_type>({ids.size()}, ids.cbegin(), ids.cend());
    }

    const std::string
    decode(index_type id) const
    {
        if (auto tok = _m_rmap.find(id); tok != _m_rmap.end()) {
            return tok->second;
        }
        throw std::runtime_error(std::format("unable to decode id '{}'", id));
    }

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
    decode(ForwardIt first, ForwardIt last)
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
