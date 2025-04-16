#pragma once

#include <concepts>
#include <ostream>
#include <span>

#include <metalchat/container.h>


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
concept immutable_tensor = requires(std::remove_reference_t<Tensor> const t) {
    /// The type of the elements.
    typename Tensor::value_type;
    typename Tensor::pointer_type;
    typename Tensor::container_type;
    typename Tensor::iterator;
    typename Tensor::const_iterator;

    { Tensor::dim() } -> std::convertible_to<std::size_t>;
    { t.sizes() } -> std::same_as<const std::span<std::size_t, Tensor::dim()>>;
    { t.strides() } -> std::same_as<const std::span<std::size_t, Tensor::dim()>>;
    { t.offsets() } -> std::same_as<const std::span<std::size_t, Tensor::dim()>>;
    { t.numel() } -> std::same_as<std::size_t>;

    { t.container() } -> std::same_as<typename Tensor::container_type&>;
    { t.layout() } -> std::same_as<tensor_layout<Tensor::dim()>>;

    { t.narrow(std::size_t(), std::size_t(), std::size_t()) } -> std::same_as<Tensor>;

    { t.data_ptr() } -> std::same_as<typename Tensor::pointer_type>;
    { t.begin() } -> std::same_as<typename Tensor::const_iterator>;
    { t.end() } -> std::same_as<typename Tensor::const_iterator>;
};


template <typename Tensor>
concept immutable_scalar = immutable_tensor<Tensor> && Tensor::dim() == 0;


template <typename Tensor>
concept immutable_tensor1d = immutable_tensor<Tensor> && Tensor::dim() == 1;


template <typename Tensor>
concept immutable_tensor2d = immutable_tensor<Tensor> && Tensor::dim() == 2;


template <typename Tensor>
concept immutable_tensor3d = immutable_tensor<Tensor> && Tensor::dim() == 3;


template <typename Tensor>
concept immutable_tensor4d = immutable_tensor<Tensor> && Tensor::dim() == 4;


template <typename Tensor>
concept immutable_device_tensor
    = immutable_tensor<Tensor>
      && std::same_as<typename Tensor::container_type, device_ref<typename Tensor::value_type>>;


template <typename Tensor>
concept immutable_device_tensor4d
    = immutable_tensor4d<Tensor>
      && std::same_as<typename Tensor::container_type, device_ref<typename Tensor::value_type>>;


} // namespace metalchat
