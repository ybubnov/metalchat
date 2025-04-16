#pragma once

#include <iostream>


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
    os.precision(3);
    os.setf(std::ios::showpos | std::ios::fixed);
    operator<< <__fp16>(os, vec);
    return os;
}
