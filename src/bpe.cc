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


re3_iterator::re3_iterator(const std::shared_ptr<pcre2_code> re, const std::string& input)
: _m_re(re),
  _m_data(pcre2_match_data_create_from_pattern(re.get(), nullptr)),
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
    auto rc = pcre2_match(_m_re.get(), _m_subject, _m_subject_length, _m_offset, 0, _m_data, NULL);
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

    auto re_ptr = pcre2_compile(
        reinterpret_cast<PCRE2_SPTR>(regex.c_str()), PCRE2_ZERO_TERMINATED, 0, &error_code,
        &error_offset, nullptr
    );

    if (re_ptr == nullptr) {
        PCRE2_UCHAR message[error_buffer_size];
        pcre2_get_error_message(error_code, message, sizeof(message));

        throw std::invalid_argument(std::format("re3: invalid regular expression: {}", message));
    }

    _m_re = std::shared_ptr<pcre2_code>(re_ptr, pcre2_code_free);
}


re3_iterator
re3::begin(const std::string& input) const
{
    return re3_iterator(_m_re, input);
}

re3_iterator
re3::end() const
{
    return re3_iterator();
}


byte_pair_encoder::byte_pair_encoder(const std::filesystem::path& p)
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


} // namespace metalchat
