// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <istream>
#include <ostream>


namespace metalchat {


/// Specifies role of the message.
using rolekind = int32_t;


struct role {
    static constexpr rolekind system = 1 << 0;
    static constexpr rolekind request = 1 << 1;
    static constexpr rolekind response = 1 << 2;
    static constexpr rolekind command = 1 << 3;
    static constexpr rolekind result = 1 << 4;
};


template <typename CharT> class basic_message {
public:
    using content_type = std::basic_string<CharT>;

    basic_message(rolekind role, const content_type& content)
    : _M_role(role),
      _M_content(content)
    {}

    basic_message(rolekind role, content_type&& content)
    : _M_role(role),
      _M_content(std::move(content))
    {}

    basic_message(rolekind role)
    : _M_role(role),
      _M_content()
    {}

    basic_message(const basic_message&) = default;

    rolekind
    role() const
    {
        return _M_role;
    }

    const content_type&
    content() const
    {
        return _M_content;
    }

    static basic_message
    request(const content_type& content)
    {
        return basic_message(role::request, content);
    }

private:
    rolekind _M_role;
    content_type _M_content;
};


class basic_token_scanner {
public:
    using index_type = int32_t;

    virtual void
    reset() = 0;

    virtual bool
    scan(index_type token) = 0;

    /// The /ref basic_token_scanner default destructor.
    virtual ~basic_token_scanner() = default;
};


class match_token_scanner : public basic_token_scanner {
public:
    match_token_scanner(std::initializer_list<index_type> tokens)
    : match_token_scanner(tokens.begin(), tokens.end())
    {}

    template <std::forward_iterator ForwardIt>
    match_token_scanner(ForwardIt first, ForwardIt last)
        requires std::same_as<std::iter_value_t<ForwardIt>, index_type>
    : _M_tokens(first, last)
    {}

    void
    reset()
    {}

    bool
    scan(index_type token)
    {
        return _M_tokens.find(token) == _M_tokens.end();
    }

private:
    std::unordered_set<index_type> _M_tokens;
};


class limit_token_scanner : public basic_token_scanner {
public:
    limit_token_scanner(std::size_t lim)
    : _M_lim(lim),
      _M_scanned(0)
    {}

    void
    reset()
    {
        _M_scanned = 0;
    }

    bool
    scan(index_type token)
    {
        return (++_M_scanned) < _M_lim;
    }

private:
    std::size_t _M_lim;
    std::size_t _M_scanned;
};


template <typename LogicalOp> class composite_token_scanner : public basic_token_scanner {
public:
    using scanner_type = basic_token_scanner;
    using scanner_pointer = std::shared_ptr<scanner_type>;

    composite_token_scanner(std::initializer_list<scanner_pointer> scanners)
    : composite_token_scanner(scanners.begin(), scanners.end())
    {}

    template <std::forward_iterator ForwardIt>
    composite_token_scanner(ForwardIt first, ForwardIt last)
        requires std::same_as<std::iter_value_t<ForwardIt>, scanner_pointer>
    : _M_scanners(std::make_move_iterator(first), std::make_move_iterator(last)),
      _M_logical_op()
    {}

    /// The default \ref composite_token_scanner constructor.
    composite_token_scanner()
    : composite_token_scanner({})
    {}

    void
    reset()
    {
        for (auto& scanner : _M_scanners) {
            scanner->reset();
        }
    }

    bool
    scan(index_type token)
    {
        bool result = false;
        if (_M_scanners.size() == 0) {
            return result;
        }

        result = _M_scanners.front()->scan(token);
        for (std::size_t i = 1; i < _M_scanners.size(); i++) {
            result = _M_logical_op(result, _M_scanners[i]->scan(token));
        }
        return result;
    }

private:
    std::vector<scanner_pointer> _M_scanners;
    LogicalOp _M_logical_op;
};


template <typename Index, typename CharT> struct basic_formatter {
    using index_type = Index;
    using char_type = CharT;
    using istream_type = std::basic_istream<index_type>;
    using ostream_type = std::basic_ostream<index_type>;
    using message_type = basic_message<char_type>;

    virtual message_type
    parse(istream_type& is) = 0;

    virtual void
    format(const message_type& message, ostream_type& os) const = 0;

    /// The \ref basic_formatter virtual destructor.
    virtual ~basic_formatter() = default;
};


} // namespace metalchat
