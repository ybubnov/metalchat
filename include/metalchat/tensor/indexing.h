// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <concepts>
#include <format>
#include <optional>
#include <ostream>


namespace metalchat {

struct slice {
    std::optional<std::size_t> start;
    std::optional<std::size_t> stop;

    slice(
        std::optional<std::size_t> start_ = std::nullopt,
        std::optional<std::size_t> stop_ = std::nullopt
    )
    : start(start_),
      stop(stop_)
    {
        if (stop < start) {
            throw std::invalid_argument(std::format(
                "slice: start position {} should be lesser than stop position {}", start.value(),
                stop.value()
            ));
        }
    }

    slice(std::size_t (&&sizes)[2])
    : start(sizes[0]),
      stop(sizes[1])
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const slice& s)
    {
        auto arr = std::to_array({s.start, s.stop});
        for (auto it = arr.begin(); it != arr.end(); ++it) {
            if (*it) {
                os << it->value();
            }
            if (it != arr.end() - 1) {
                os << ":";
            }
        }
        return os;
    }
};


template <class T>
concept convertible_to_slice = std::convertible_to<T, slice>;


template <class T>
concept convertible_to_index = std::convertible_to<T, std::size_t>;


} // namespace metalchat
