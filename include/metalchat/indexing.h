#pragma once


#include <array>
#include <concepts>
#include <optional>
#include <ostream>


namespace metalchat {
namespace indexing {

struct slice {
    std::optional<std::size_t> start;
    std::optional<std::size_t> stop;

    slice(
        std::optional<std::size_t> start_,
        std::optional<std::size_t> stop_
    )
    : start(start_),
      stop(stop_)
    {
        assert(stop >= start);
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
concept SliceConvertible = std::is_convertible_v<T, slice>;


} // namespace indexing
} // namespace metalchat
