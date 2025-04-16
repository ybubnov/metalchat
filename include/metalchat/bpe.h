#pragma once

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cppcodec/base64_rfc4648.hpp>
#include <unicode/regex.h>

#include <metalchat/container.h>
#include <metalchat/tensor.h>


namespace metalchat {


class bpe {
public:
    using string_type = std::string;
    using index_type = int32_t;

private:
    using base64 = cppcodec::base64_rfc4648;

    std::unordered_map<string_type, index_type> _m_fmap;
    std::unordered_map<index_type, string_type> _m_rmap;

    std::unique_ptr<icu::RegexMatcher> _m_re;

public:
    static constexpr index_type pad = -1;

    /// A regular expression string that is used to split the input text into tokens.
    static constexpr const char* token_regex
        = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

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

    tensor<index_type, 1, owning_ref<index_type>>
    encode(const std::string& s)
    {
        _m_re->reset(icu::UnicodeString::fromUTF8(s));
        std::vector<int32_t> ids;

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
            match.toUTF8String(key);

            if (auto it = _m_fmap.find(key); it != _m_fmap.end()) {
                ids.push_back(it->second);
            }
        }

        return tensor(tensor_base<index_type, 1, owning_ref<index_type>>(std::move(ids)));
    }
};


} // namespace metalchat
