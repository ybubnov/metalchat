#pragma once

#include <concepts>
#include <ostream>
#include <span>


namespace metalchat {


template <uint32_t N> struct tensor_layout {
    uint32_t sizes[N];
    uint32_t strides[N];
    uint32_t offsets[N];

    friend std::ostream&
    operator<<(std::ostream& os, const tensor_layout& l)
    {
        os << "layout<" << N << ">{sizes=[";
        for (auto i = 0; i < N; i++) {
            os << l.sizes[i];
            if (i < N - 1) {
                os << ",";
            }
        }
        os << "]}";
        return os;
    }
};


template <typename Tensor>
concept is_tensor = requires(std::remove_reference_t<Tensor> const t) {
    // typename Tensor::dimensions;
    typename Tensor::value_type;
    typename Tensor::pointer_type;
    typename Tensor::container_type;

    { Tensor::dim() } -> std::convertible_to<std::size_t>;
    { t.sizes() } -> std::same_as<const std::span<std::size_t>>;
    { t.strides() } -> std::same_as<const std::span<std::size_t>>;
    { t.offsets() } -> std::same_as<const std::span<std::size_t>>;
    { t.numel() } -> std::same_as<std::size_t>;
    { t.data_ptr() } -> std::same_as<typename Tensor::pointer_type>;
    //{ t.layout() } -> std::same_as<tensor_layout<Tensor::dimensions>>;
};


} // namespace metalchat
