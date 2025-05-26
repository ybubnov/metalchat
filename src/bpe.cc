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


struct re3::impl {
    std::shared_ptr<pcre2_code> ptr = nullptr;
};


re3::re3(const std::string& regex)
{
    int error_code;
    PCRE2_SIZE error_offset;

    auto re_ptr = pcre2_compile(
        reinterpret_cast<PCRE2_SPTR>(regex.c_str()), PCRE2_ZERO_TERMINATED, 0, &error_code,
        &error_offset, nullptr
    );

    if (re_ptr == nullptr) {
        PCRE2_UCHAR message[error_buffer_size];
        pcre2_get_error_message(error_code, message, sizeof(message));

        throw std::invalid_argument(std::format("re3: invalid regular expression: {}", message));
    }

    auto ptr = std::shared_ptr<pcre2_code>(re_ptr, pcre2_code_free);
    _m_impl = std::make_shared<impl>(ptr);
}


re3::iterator
re3::begin(const std::string& input) const
{
    return re3::iterator(_m_impl, input);
}


re3::iterator
re3::end() const
{
    return re3::iterator();
}


struct re3::iterator::impl {
    const std::shared_ptr<pcre2_code> _m_re = nullptr;
    pcre2_match_data* _m_data = nullptr;
    const PCRE2_SPTR _m_subject = nullptr;
    const PCRE2_SIZE _m_subject_length = 0;
    PCRE2_SIZE _m_offset = 0;
    bool _m_end = false;

    impl()
    : _m_re(nullptr),
      _m_data(nullptr),
      _m_subject(nullptr),
      _m_subject_length(0),
      _m_offset(0),
      _m_end(true)
    {}

    impl(const std::shared_ptr<re3::impl> re_impl, const std::string& input)
    : _m_re(re_impl->ptr),
      _m_data(pcre2_match_data_create_from_pattern(re_impl->ptr.get(), nullptr)),
      _m_subject(reinterpret_cast<PCRE2_SPTR>(input.c_str())),
      _m_subject_length(input.size()),
      _m_offset(0),
      _m_end(false)
    {}

    ~impl()
    {
        if (_m_data != nullptr) {
            pcre2_match_data_free(_m_data);
        }
    }
};


re3::iterator::iterator()
: _m_impl(std::make_shared<re3::iterator::impl>())
{}


re3::iterator::iterator(const std::shared_ptr<re3::impl> re_impl, const std::string& input)
: _m_impl(std::make_shared<re3::iterator::impl>(re_impl, input))
{
    next();
}


re3::iterator&
re3::iterator::operator++()
{
    if (!_m_impl->_m_end) {
        next();
    }
    return *this;
}


re3::iterator::value_type
re3::iterator::operator*()
{
    return get();
}


bool
re3::iterator::operator!=(const iterator& rhs)
{
    return _m_impl->_m_end != rhs._m_impl->_m_end;
}


re3::iterator::value_type
re3::iterator::get()
{
    if (_m_impl->_m_end) {
        throw std::runtime_error(std::format("re3_iterator: terminated iterator cannot be accessed")
        );
    }

    PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(_m_impl->_m_data);
    PCRE2_SIZE length = ovector[1] - ovector[0];

    _m_impl->_m_offset = ovector[1];

    value_type result(reinterpret_cast<const char*>(_m_impl->_m_subject + ovector[0]), length);
    return result;
}


void
re3::iterator::next()
{
    auto rc = pcre2_match(
        _m_impl->_m_re.get(), _m_impl->_m_subject, _m_impl->_m_subject_length, _m_impl->_m_offset,
        0, _m_impl->_m_data, NULL
    );
    if (rc < 0) {
        _m_impl->_m_end = true;
        if (rc != PCRE2_ERROR_NOMATCH) {
            throw std::runtime_error(std::format("re3_iterator: matching error {}", rc));
        }
    }
}


} // namespace metalchat
