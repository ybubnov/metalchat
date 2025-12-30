// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <format>
#include <iterator>
#include <memory>
#include <string>


namespace metalchat {
namespace text {


class regexp_iterator;


class regexp {
public:
    regexp(const std::string& regex);
    regexp(const char* regex);

    regexp_iterator
    begin(const std::string&) const;

    regexp_iterator
    end() const;

private:
    struct _RegularExpression;
    std::shared_ptr<_RegularExpression> _M_impl;

    friend class regexp_iterator;
};


/// Regular expression iterator.
///
/// This iterator is used to provide convenient interface to access match group
/// data. So every match is considered an element of the backing container and
/// this iterator returns matches sequentially until the last match.
class regexp_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::string;
    using reference = value_type&;
    using pointer = value_type*;
    using difference_type = std::ptrdiff_t;

    /// Advance the iterator to the next regular expression match.
    regexp_iterator&
    operator++();

    /// Return the current match of the regular expression.
    ///
    /// The method throws `std::runtime_error` when it is called on a terminated iterator.
    value_type
    operator*();

    /// Compares two regular expression iterators.
    ///
    /// The implementation is naive for simplicity reasons, and only compares
    /// the ends of iterators.
    bool
    operator!=(const regexp_iterator&);

    /// Initialize the end-of-match-group iterator.
    regexp_iterator();

    /// Initializes the iterators, stores the address of `regexp` in data member, and performs
    /// the finds the first match from the input string to initialize match group members.
    regexp_iterator(const regexp& regex, const std::string& input);

private:
    struct _RegularExpressionIterator;
    std::shared_ptr<_RegularExpressionIterator> _M_impl;

    /// Advance the iterator to the next match group.
    void
    next();

    /// Get the current match group value.
    value_type
    get();

    friend class regexp;
};


class base64 {
public:
    static std::string
    decode(const std::string&);
};


} // namespace text
} // namespace metalchat
