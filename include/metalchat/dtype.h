// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <array>
#include <bit>
#include <cfloat>
#include <concepts>
#include <iostream>


namespace metalchat {


struct bf16 {
    using bits_type = std::array<uint16_t, 2>;

    uint16_t bits = 0;

    bf16(float f) { (*this) = f; }

    bf16() = default;

    operator float() const
    {
        bits_type float_bits = {{0, bits}};
        return std::bit_cast<float>(float_bits);
    }

    bf16&
    operator=(float f)
    {
        auto float_bits = std::bit_cast<bits_type>(f);

        switch (std::fpclassify(f)) {
        case FP_SUBNORMAL:
        case FP_ZERO:
            bits = float_bits[1];
            bits &= 0x8000;
            break;
        case FP_INFINITE:
            bits = float_bits[1];
            break;
        case FP_NAN:
            bits = float_bits[1];
            bits |= 1 << 6;
            break;
        case FP_NORMAL:
            const uint32_t rounding_bias = 0x00007FFF + (float_bits[1] & 0x1);
            const uint32_t int_bits = std::bit_cast<uint32_t>(f) + rounding_bias;
            float_bits = std::bit_cast<bits_type>(int_bits);
            bits = float_bits[1];
            break;
        }
        return *this;
    }

    friend std::ostream&
    operator<<(std::ostream& os, const bf16& f)
    {
        os << float(f);
        return os;
    }

    bf16
    operator*=(const bf16& f)
    {
        (*this) = float{*this} * f;
        return *this;
    }

    bf16
    operator+=(const bf16& f)
    {
        (*this) = float{*this} + f;
        return *this;
    }
};


template <typename T> struct type_traits;

template <> struct type_traits<bf16> {
    static std::string
    name()
    {
        return "bfloat";
    }
};


template <> struct type_traits<float> {
    static std::string
    name()
    {
        return "float";
    }
};


template <> struct type_traits<int32_t> {
    static std::string
    name()
    {
        return "int32_t";
    }
};


template <> struct type_traits<int8_t> {
    static std::string
    name()
    {
        return "int8_t";
    }
};


} // namespace metalchat
