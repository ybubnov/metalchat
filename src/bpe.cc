#define PCRE2_CODE_UNIT_WIDTH 8

#include <cppcodec/base64_rfc4648.hpp>
#include <pcre2.h>

#include <metalchat/bpe.h>


namespace metalchat {


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
        reinterpret_cast<PCRE2_SPTR>(regex.c_str()), PCRE2_ZERO_TERMINATED, 0, &error_code,
        &error_offset, nullptr
    );

    if (re_ptr == nullptr) {
        PCRE2_UCHAR message[_RegularExpression_error_bufsize];
        pcre2_get_error_message(error_code, message, sizeof(message));

        throw std::invalid_argument(std::format("regexp: invalid regular expression: {}", message));
    }

    auto ptr = std::shared_ptr<pcre2_code>(re_ptr, pcre2_code_free);
    _m_impl = std::make_shared<_RegularExpression>(ptr);
}


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

    const std::shared_ptr<pcre2_code> _m_re = nullptr;
    pcre2_match_data* _m_data = nullptr;
    const PCRE2_SPTR _m_subject = nullptr;
    const PCRE2_SIZE _m_subject_length = 0;
    PCRE2_SIZE _m_offset = 0;
    bool _m_end = false;

    _RegularExpressionIterator()
    : _m_re(nullptr),
      _m_data(nullptr),
      _m_subject(nullptr),
      _m_subject_length(0),
      _m_offset(0),
      _m_end(true)
    {}

    _RegularExpressionIterator(const regexp& regex, const std::string& input)
    : _m_re(regex._m_impl->ptr),
      _m_data(pcre2_match_data_create_from_pattern(regex._m_impl->ptr.get(), nullptr)),
      _m_subject(reinterpret_cast<PCRE2_SPTR>(input.c_str())),
      _m_subject_length(input.size()),
      _m_offset(0),
      _m_end(false)
    {}

    ~_RegularExpressionIterator()
    {
        if (_m_data != nullptr) {
            pcre2_match_data_free(_m_data);
        }
    }
};


regexp_iterator::regexp_iterator()
: _m_impl(std::make_shared<regexp_iterator::_RegularExpressionIterator>())
{}


regexp_iterator::regexp_iterator(const regexp& regex, const std::string& input)
: _m_impl(std::make_shared<regexp_iterator::_RegularExpressionIterator>(regex, input))
{
    next();
}


regexp_iterator&
regexp_iterator::operator++()
{
    if (!_m_impl->_m_end) {
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
    return _m_impl->_m_end != rhs._m_impl->_m_end;
}


regexp_iterator::value_type
regexp_iterator::get()
{
    if (_m_impl->_m_end) {
        throw std::runtime_error(
            std::format("regexp_iterator: terminated iterator cannot be accessed")
        );
    }

    PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(_m_impl->_m_data);
    PCRE2_SIZE length = ovector[1] - ovector[0];

    _m_impl->_m_offset = ovector[1];

    value_type result(reinterpret_cast<const char*>(_m_impl->_m_subject + ovector[0]), length);
    return result;
}


void
regexp_iterator::next()
{
    auto rc = pcre2_match(
        _m_impl->_m_re.get(), _m_impl->_m_subject, _m_impl->_m_subject_length, _m_impl->_m_offset,
        0, _m_impl->_m_data, NULL
    );
    if (rc < 0) {
        _m_impl->_m_end = true;
        if (rc != PCRE2_ERROR_NOMATCH) {
            throw std::runtime_error(std::format("regexp_iterator: matching error {}", rc));
        }
    }
}


} // namespace metalchat
