#pragma once
#define PCRE2_CODE_UNIT_WIDTH 8

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
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

    re3_iterator(pcre2_code* re, const std::string& input)
    : _m_re(re),
      _m_data(pcre2_match_data_create_from_pattern(re, nullptr)),
      _m_subject(reinterpret_cast<PCRE2_SPTR>(input.c_str())),
      _m_subject_length(input.size()),
      _m_offset(0),
      _m_end(false)
    {
        next();
    }

    re3_iterator()
    : _m_re(nullptr),
      _m_data(nullptr),
      _m_subject(nullptr),
      _m_subject_length(0),
      _m_offset(0),
      _m_end(true)
    {}

    ~re3_iterator()
    {
        if (_m_data != nullptr) {
            pcre2_match_data_free(_m_data);
        }
    }

    iterator&
    operator++()
    {
        if (!_m_end) {
            next();
        }
        return *this;
    }

    value_type
    operator*()
    {
        return get();
    }

    bool
    operator!=(const iterator& rhs)
    {
        return _m_end != rhs._m_end;
    }

private:
    pcre2_code* _m_re = nullptr;
    pcre2_match_data* _m_data = nullptr;
    const PCRE2_SPTR _m_subject = nullptr;
    const PCRE2_SIZE _m_subject_length = 0;
    PCRE2_SIZE _m_offset = 0;
    bool _m_end = false;

    value_type
    get()
    {
        if (_m_end) {
            throw std::runtime_error(
                std::format("re3_iterator: terminated iterator cannot be accessed")
            );
        }

        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(_m_data);
        PCRE2_SIZE length = ovector[1] - ovector[0];

        _m_offset = ovector[1];

        value_type result(reinterpret_cast<const char*>(_m_subject + ovector[0]), length);
        return result;
    }

    void
    next()
    {
        auto rc = pcre2_match(_m_re, _m_subject, _m_subject_length, _m_offset, 0, _m_data, NULL);
        if (rc < 0) {
            _m_end = true;
            if (rc != PCRE2_ERROR_NOMATCH) {
                throw std::runtime_error(std::format("re3_iterator: matching error {}", rc));
            }
        }
    }
};


class re3 {
private:
    pcre2_code* _m_re = nullptr;

    static constexpr std::size_t error_buffer_size = 256;

public:
    re3(const std::string& regex)
    {
        int error_code;
        PCRE2_SIZE error_offset;

        _m_re = pcre2_compile(
            reinterpret_cast<PCRE2_SPTR>(regex.c_str()), PCRE2_ZERO_TERMINATED, 0, &error_code,
            &error_offset, nullptr
        );

        if (_m_re == nullptr) {
            PCRE2_UCHAR message[error_buffer_size];
            pcre2_get_error_message(error_code, message, sizeof(message));

            throw std::invalid_argument(std::format("re3: invalid regular expression: {}", message)
            );
        }
    }

    re3_iterator
    begin(const std::string& input)
    {
        return re3_iterator(_m_re, input);
    }

    re3_iterator
    end()
    {
        return re3_iterator();
    }

    ~re3()
    {
        if (_m_re != nullptr) {
            pcre2_code_free(_m_re);
        }
    }
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

    re3 _m_re;

    /// A regular expression string that is used to split the input text into tokens.
    static constexpr const char* token_regex = (R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|)"
                                                R"([^\r\n\p{L}\p{N}]?\p{L}+|)"
                                                R"(\p{N}{1,3}|)"
                                                R"( ?[^\s\p{L}\p{N}]+[\r\n]*|)"
                                                R"(\s*[\r\n]+|)"
                                                R"(\s+(?!\S)|)"
                                                R"(\s+)");

public:
    static constexpr index_type pad = -1;

    /// Number of special tokens used to prepare the input for the model.
    static constexpr const std::size_t nspecial = 256;

    bpe(const std::filesystem::path& p)
    : _m_fmap(),
      _m_rmap(),
      _m_re(token_regex)
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
            string_type value = base64::decode<string_type>(key_part);

            _m_fmap.insert(std::make_pair(value, key));
            _m_rmap.insert(std::make_pair(key, value));
        }
    }

    template <push_back_container PushBackContainer>
    void
    encode(const std::string& s, PushBackContainer& ids)
    {
        for (auto match = _m_re.begin(s); match != _m_re.end(); ++match) {
            auto key = (*match);
            if (auto it = _m_fmap.find(key); it != _m_fmap.end()) {
                ids.push_back(it->second);
            } else {
                throw std::runtime_error(
                    std::format("bpe: key '{}'(size={}) is missing", key, key.size())
                );
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
            throw std::invalid_argument(std::format("bpe: unknown special token '{}'", index));
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
        throw std::runtime_error(std::format("bpe: unable to decode id '{}'", id));
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
