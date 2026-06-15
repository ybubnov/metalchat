// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#define PCRE2_CODE_UNIT_WIDTH 8

#include <cppcodec/base64_rfc4648.hpp>
#include <pcre2.h>

#include <metalchat/text/regexp.h>


namespace metalchat {
namespace text {


std::string
base64::decode(const std::string& s)
{
    return cppcodec::base64_rfc4648::decode<std::string>(s);
}


static constexpr std::size_t _RegularExpression_error_bufsize = 256;


struct regexp::_RegularExpression {
    std::shared_ptr<pcre2_code> ptr = nullptr;
};


regexp::regexp(const std::string& regex)
{
    int error_code;
    PCRE2_SIZE error_offset;

    auto re_ptr = pcre2_compile(
        reinterpret_cast<PCRE2_SPTR>(regex.data()), regex.size(), 0, &error_code, &error_offset,
        nullptr
    );

    if (re_ptr == nullptr) {
        PCRE2_UCHAR message[_RegularExpression_error_bufsize];
        pcre2_get_error_message(error_code, message, sizeof(message));

        throw std::invalid_argument(std::format("regexp: invalid regular expression: {}", message));
    }

    auto ptr = std::shared_ptr<pcre2_code>(re_ptr, pcre2_code_free);
    _M_impl = std::make_shared<_RegularExpression>(ptr);
}


regexp::regexp(const char* regex)
: regexp(std::string(regex))
{}


regexp_iterator
regexp::begin(const std::string& input) const
{
    return regexp_iterator(*this, input);
}


regexp_iterator
regexp::end() const
{
    return regexp_iterator();
}


struct regexp_iterator::_RegularExpressionIterator {
    friend class regexp;

    const std::shared_ptr<pcre2_code> _M_re = nullptr;
    std::shared_ptr<pcre2_match_data> _M_data = nullptr;
    std::string _M_subject;
    PCRE2_SIZE _M_offset = 0;
    bool _M_end = false;

    _RegularExpressionIterator()
    : _M_re(nullptr),
      _M_data(nullptr),
      _M_subject(),
      _M_offset(0),
      _M_end(true)
    {}

    _RegularExpressionIterator(const regexp& regex, const std::string& input)
    : _M_re(regex._M_impl->ptr),
      _M_data(nullptr),
      _M_subject(input),
      _M_offset(0),
      _M_end(false)
    {
        pcre2_code* code = regex._M_impl->ptr.get();
        pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(code, nullptr);

        _M_data = std::shared_ptr<pcre2_match_data>(match_data, pcre2_match_data_free);
        _M_end = pcre2_get_ovector_count(match_data) == 0;
    }
};


regexp_iterator::regexp_iterator()
: _M_impl(std::make_shared<regexp_iterator::_RegularExpressionIterator>())
{}


regexp_iterator::regexp_iterator(const regexp& regex, const std::string& input)
: _M_impl(std::make_shared<regexp_iterator::_RegularExpressionIterator>(regex, input))
{
    next();
}


regexp_iterator&
regexp_iterator::operator++()
{
    if (!_M_impl->_M_end) {
        next();
    }
    return *this;
}


regexp_iterator::value_type
regexp_iterator::operator*()
{
    return get();
}


bool
regexp_iterator::operator!=(const regexp_iterator& rhs)
{
    return _M_impl->_M_end != rhs._M_impl->_M_end;
}


regexp_iterator::value_type
regexp_iterator::get()
{
    if (_M_impl->_M_end) {
        throw std::runtime_error(
            std::format("regexp_iterator: terminated iterator cannot be accessed")
        );
    }

    pcre2_match_data* match_data = _M_impl->_M_data.get();

    PCRE2_SIZE* slice = pcre2_get_ovector_pointer(match_data);
    PCRE2_SIZE length = slice[1] - slice[0];

    auto data = _M_impl->_M_subject.data() + _M_impl->_M_offset;
    _M_impl->_M_offset = slice[1];
    _M_impl->_M_end |= _M_impl->_M_subject.size() == _M_impl->_M_offset;

    return value_type(reinterpret_cast<const char*>(data), length);
}


void
regexp_iterator::next()
{
    pcre2_match_data* match_data = _M_impl->_M_data.get();
    const PCRE2_SPTR subject = reinterpret_cast<PCRE2_SPTR>(_M_impl->_M_subject.data());
    const PCRE2_SIZE subject_size = _M_impl->_M_subject.size();

    auto rc = pcre2_match(
        _M_impl->_M_re.get(), subject, subject_size, _M_impl->_M_offset, 0, match_data, NULL
    );
    if (rc < 0) {
        _M_impl->_M_end = true;
        if (rc != PCRE2_ERROR_NOMATCH) {
            throw std::runtime_error(std::format("regexp_iterator: matching error {}", rc));
        }
    }
}


} // namespace text
} // namespace metalchat
