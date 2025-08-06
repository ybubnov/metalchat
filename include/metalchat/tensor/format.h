#pragma once

#include <format>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

#include <metalchat/tensor/basic.h>
#include <metalchat/tensor/future.h>
#include <metalchat/tensor/shared.h>


namespace metalchat {
namespace fmt {

constexpr std::size_t precision = 3;
constexpr std::size_t edgeitems = 80;


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


template <typename T, std::size_t N, contiguous_container Container> struct basic_tensor_format {
    const tensor<T, N, Container>& t;
    const int w;

    basic_tensor_format(const tensor<T, N, Container>& t_, const int w_ = 0)
    : t(t_),
      w(w_)
    {}
};


template <typename T, std::size_t N, contiguous_container Container>
struct tensor_format : public basic_tensor_format<T, N, Container> {
    tensor_format(const tensor<T, N, Container>& tensor, const int w = 0)
    : basic_tensor_format<T, N, Container>(tensor, w)
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const tensor_format<T, N, Container>& tf)
    {
        auto size = tf.t.size(0);
        auto max_size = fmt::edgeitems * 2 + 1;

        using format_type = tensor_format<T, N - 1, reference_memory_container<T>>;

        os << "[";
        if (size > max_size) {
            for (std::size_t i = 0; i < fmt::edgeitems; i++) {
                os << format_type(tf.t.at(i), tf.w + 1) << fmt::comma(i, size);
                os << std::endl << std::setw(tf.w) << "";
            }

            os << "..., " << std::endl << std::setw(tf.w) << "";

            for (std::size_t i = size - fmt::edgeitems; i < size; i++) {
                os << format_type(tf.t.at(i), tf.w + 1) << fmt::comma(i, size);
                if (i < size - 1) {
                    os << std::endl << std::setw(tf.w) << "";
                }
            }
        } else {
            for (std::size_t i = 0; i < size; i++) {
                os << format_type(tf.t.at(i), tf.w + 1) << fmt::comma(i, size);
                if (i < size - 1) {
                    os << std::endl << std::setw(tf.w) << "";
                }
            }
        }
        os << "]";
        return os;
    }
};


template <typename T, contiguous_container Container>
struct tensor_format<T, 1, Container> : public basic_tensor_format<T, 1, Container> {
    tensor_format(const tensor<T, 1, Container>& tensor, const int w = 0)
    : basic_tensor_format<T, 1, Container>(tensor, w)
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const tensor_format<T, 1, Container>& tf)
    {
        auto size = tf.t.size(0);
        auto max_size = fmt::edgeitems * 2 + 1;

        using format_type = tensor_format<T, 0, reference_memory_container<T>>;

        os << "[";
        if (size > max_size) {
            for (std::size_t i = 0; i < fmt::edgeitems; i++) {
                os << format_type(tf.t.at(i)) << fmt::comma(i, size);
            }
            os << " ..., ";
            for (std::size_t i = size - fmt::edgeitems; i < size; i++) {
                os << format_type(tf.t.at(i)) << fmt::comma(i, size);
            }
        } else {
            for (std::size_t i = 0; i < size; i++) {
                os << format_type(tf.t.at(i)) << fmt::comma(i, size);
            }
        }
        os << "]";
        return os;
    }
};


template <typename T, contiguous_container Container>
struct tensor_format<T, 0, Container> : public basic_tensor_format<T, 0, Container> {
    tensor_format(const tensor<T, 0, Container>& tensor, const int w = 0)
    : basic_tensor_format<T, 0, Container>(tensor, w)
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const tensor_format<T, 0, Container>& tf)
    {
        os << *tf.t.data_ptr();
        return os;
    }
};


template <typename T, std::size_t N>
std::ostream&
operator<<(std::ostream& os, const std::span<T, N>& arr)
{
    for (auto it = arr.begin(); it != arr.end(); ++it) {
        os << *it;
        if (it != arr.end() - 1) {
            os << ", ";
        }
    }
    return os;
}


template <typename T, std::size_t N, contiguous_container Container>
std::ostream&
operator<<(std::ostream& os, const tensor<T, N, Container>& t)
{
    os << tensor_format<T, N, Container>(t, 1) << ", sizes=(" << t.sizes() << ")";
    return os;
}


template <typename T, std::size_t N, contiguous_container Container>
std::ostream&
operator<<(std::ostream& os, const shared_tensor<T, N, Container>& t)
{
    os << (*t) << ", shared=true";
    return os;
}


template <typename T, std::size_t N>
std::ostream&
operator<<(std::ostream& os, const future_tensor<T, N>& t)
{
    os << t.get() << ", future=true";
    return os;
}


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


} // namespace metalchat
