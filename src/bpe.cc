#include <metalchat/bpe.h>


namespace metalchat {


re3_iterator::re3_iterator()
: _m_re(nullptr),
  _m_data(nullptr),
  _m_subject(nullptr),
  _m_subject_length(0),
  _m_offset(0),
  _m_end(true)
{}


re3_iterator::re3_iterator(pcre2_code* re, const std::string& input)
: _m_re(re),
  _m_data(pcre2_match_data_create_from_pattern(re, nullptr)),
  _m_subject(reinterpret_cast<PCRE2_SPTR>(input.c_str())),
  _m_subject_length(input.size()),
  _m_offset(0),
  _m_end(false)
{
    next();
}

re3_iterator::~re3_iterator()
{
    if (_m_data != nullptr) {
        pcre2_match_data_free(_m_data);
    }
}

re3_iterator::iterator&
re3_iterator::operator++()
{
    if (!_m_end) {
        next();
    }
    return *this;
}


re3_iterator::value_type
re3_iterator::operator*()
{
    return get();
}


bool
re3_iterator::operator!=(const re3_iterator::iterator& rhs)
{
    return _m_end != rhs._m_end;
}


re3_iterator::value_type
re3_iterator::get()
{
    if (_m_end) {
        throw std::runtime_error(std::format("re3_iterator: terminated iterator cannot be accessed")
        );
    }

    PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(_m_data);
    PCRE2_SIZE length = ovector[1] - ovector[0];

    _m_offset = ovector[1];

    value_type result(reinterpret_cast<const char*>(_m_subject + ovector[0]), length);
    return result;
}


void
re3_iterator::next()
{
    auto rc = pcre2_match(_m_re, _m_subject, _m_subject_length, _m_offset, 0, _m_data, NULL);
    if (rc < 0) {
        _m_end = true;
        if (rc != PCRE2_ERROR_NOMATCH) {
            throw std::runtime_error(std::format("re3_iterator: matching error {}", rc));
        }
    }
}


re3::re3(const std::string& regex)
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

        throw std::invalid_argument(std::format("re3: invalid regular expression: {}", message));
    }
}

re3::~re3()
{
    if (_m_re != nullptr) {
        pcre2_code_free(_m_re);
    }
}


re3_iterator
re3::begin(const std::string& input)
{
    return re3_iterator(_m_re, input);
}

re3_iterator
re3::end()
{
    return re3_iterator();
}


} // namespace metalchat
