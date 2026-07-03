// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <istream>
#include <list>
#include <ostream>
#include <unordered_set>


namespace metalchat {


/// Specifies role of the message.
using rolekind = int32_t;


struct role {
    static constexpr rolekind undefined = 1 << 0;
    static constexpr rolekind system = 1 << 1;
    static constexpr rolekind request = 1 << 2;
    static constexpr rolekind response = 1 << 3;
    static constexpr rolekind command = 1 << 4;
    static constexpr rolekind result = 1 << 5;
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
    system(const content_type& content)
    {
        return basic_message(role::system, content);
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


using message = basic_message<char>;


template <typename Index> class basic_token_scanner {
public:
    using index_type = Index;

    /// Rests the state of a scanner.
    ///
    /// The default implementation does nothing.
    virtual void
    reset() {};

    virtual bool
    scan(index_type token) = 0;

    /// The \ref basic_token_scanner default destructor.
    virtual ~basic_token_scanner() = default;
};


template <typename Index> class match_token_scanner : public basic_token_scanner<Index> {
public:
    using index_type = Index;

    match_token_scanner(std::initializer_list<index_type> tokens)
    : match_token_scanner(tokens.begin(), tokens.end())
    {}

    template <std::forward_iterator ForwardIt>
    match_token_scanner(ForwardIt first, ForwardIt last)
        requires std::same_as<std::iter_value_t<ForwardIt>, index_type>
    : _M_tokens(first, last)
    {}

    /// Does nothing, \ref match_token_scanner is stateless.
    void
    reset() override
    {}

    bool
    scan(index_type token)
    {
        return _M_tokens.find(token) == _M_tokens.end();
    }

private:
    std::unordered_set<index_type> _M_tokens;
};


template <typename Index> class limit_token_scanner : public basic_token_scanner<Index> {
public:
    using index_type = Index;

    limit_token_scanner(std::size_t lim)
    : _M_lim(lim),
      _M_scanned(0)
    {}

    /// Resets the number of tokens scanned since the last reset.
    void
    reset() override
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


template <typename Index, typename LogicalOp>
class composite_token_scanner : public basic_token_scanner<Index> {
public:
    using index_type = Index;
    using scanner_type = basic_token_scanner<index_type>;
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
    push_front(const scanner_pointer& ptr)
    {
        _M_scanners.push_front(ptr);
    }

    void
    push_back(const scanner_pointer& ptr)
    {
        _M_scanners.push_back(ptr);
    }

    /// Resets the states of all underlying token scallers.
    void
    reset() override
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

        auto scanner_it = _M_scanners.begin();
        result = (*scanner_it)->scan(token);
        for (; scanner_it != _M_scanners.end(); ++scanner_it) {
            result = _M_logical_op(result, (*scanner_it)->scan(token));
        }
        return result;
    }

private:
    std::list<scanner_pointer> _M_scanners;
    LogicalOp _M_logical_op;
};


template <std::input_iterator InputIt, typename Index = std::iter_value_t<InputIt>>
auto
make_default_scanner(
    InputIt first, InputIt last, std::optional<std::size_t> max_length = std::nullopt
)
{
    using CompositeScanner = composite_token_scanner<Index, std::logical_and<bool>>;
    using TokenScanner = match_token_scanner<Index>;
    using LimitScanner = limit_token_scanner<Index>;

    CompositeScanner scanner({std::make_shared<TokenScanner>(first, last)});
    if (max_length) {
        scanner.push_front(std::make_shared<LimitScanner>(max_length.value()));
    }
    return scanner;
}


template <typename Index, typename CharT> struct basic_formatter {
    using index_type = Index;
    using char_type = CharT;
    using istream_type = std::basic_istream<index_type>;
    using ostream_type = std::basic_ostream<index_type>;
    using message_type = basic_message<char_type>;

    virtual message_type
    parse(istream_type& is) = 0;

    virtual void
    parse(istream_type& is, std::basic_ostream<char_type>& os) = 0;

    virtual void
    format(const message_type& message, ostream_type& os) = 0;

    /// The \ref basic_formatter virtual destructor.
    virtual ~basic_formatter() = default;
};


} // namespace metalchat
