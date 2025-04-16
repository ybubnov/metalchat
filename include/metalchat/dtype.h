#pragma once

#include <array>
#include <bit>
#include <cfloat>
#include <concepts>
#include <iostream>


namespace metalchat {


template <typename T>
concept integral = std::is_integral_v<T>;


namespace dtype {


struct bf16 {
    using bits_type = std::array<uint16_t, 2>;

    uint16_t m_bits;

    bf16(float f) { (*this) = f; }

    bf16() = default;

    operator float() const
    {
        bits_type bits = {{0, m_bits}};
        return std::bit_cast<float>(bits);
    }

    bf16&
    operator=(float f)
    {
        auto bits = std::bit_cast<bits_type>(f);

        switch (std::fpclassify(f)) {
        case FP_SUBNORMAL:
        case FP_ZERO:
            m_bits = bits[1];
            m_bits &= 0x8000;
            break;
        case FP_INFINITE:
            m_bits = bits[1];
            break;
        case FP_NAN:
            m_bits = bits[1];
            m_bits |= 1 << 6;
            break;
        case FP_NORMAL:
            const uint32_t rounding_bias = 0x00007FFF + (bits[1] & 0x1);
            const uint32_t int_bits = std::bit_cast<uint32_t>(f) + rounding_bias;
            bits = std::bit_cast<bits_type>(int_bits);
            m_bits = bits[1];
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


} // namespace dtype


template <typename T> struct type_traits;

template <> struct type_traits<dtype::bf16> {
    static std::string
    name()
    {
        return "bf16";
    }
};


template <> struct type_traits<float> {
    static std::string
    name()
    {
        return "float";
    }
};


} // namespace metalchat
