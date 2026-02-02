// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <iterator>


namespace metalchat {
namespace runtime {


template <typename F> class function_output_iterator {
public:
    using iterator_category = std::output_iterator_tag;
    using value_type = void;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = void;

    function_output_iterator(F&& f)
    : _M_func(std::move(f))
    {}

    template <typename... Args>
    function_output_iterator&
    operator=(Args&&... args)
    {
        _M_func(std::forward<Args>(args)...);
        return *this;
    }

    function_output_iterator&
    operator*()
    {
        return *this;
    }

    function_output_iterator&
    operator++()
    {
        return *this;
    }

    function_output_iterator&
    operator++(int)
    {
        return *this;
    }

private:
    F _M_func;
};


} // namespace runtime
} // namespace metalchat
