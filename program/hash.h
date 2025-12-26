// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once


#include <functional>
#include <tuple>


template <typename SizeT, typename T>
void
hash_combine(SizeT& seed, T value)
{
    std::hash<T> make_hash;
    seed ^= make_hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}


template <typename T> struct std::hash<std::tuple<T, T>> {
    using argument_type = std::tuple<T, T>;

    size_t
    operator()(const argument_type& argument) const noexcept
    {
        size_t seed = 0;
        hash_combine(seed, std::get<0>(argument));
        hash_combine(seed, std::get<1>(argument));
        return seed;
    }
};
