#pragma once

#include <format>
#include <iostream>


namespace metalama {
namespace fmt {

constexpr int precision = 3;
constexpr int edgeitems = 3;


struct comma {
    std::size_t i;
    std::size_t size;

    comma(std::size_t i_, std::size_t size_)
    : i(i_),
      size(size_)
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const comma& c)
    {
        if (c.i < c.size - 1) {
            os << ",";
        }
        return os;
    }
};


} // namespace fmt


template <typename T>
std::ostream&
operator<<(std::ostream& os, const std::vector<T>& vec)
{
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        os << *it;
        if (it != vec.end() - 1) {
            os << ", ";
        }
    }
    return os;
}


std::ostream&
operator<<(std::ostream& os, const std::vector<__fp16>& vec)
{
    os.precision(fmt::precision);
    os.setf(std::ios::showpos | std::ios::fixed);
    operator<< <__fp16>(os, vec);
    return os;
}


} // namespace metalama
