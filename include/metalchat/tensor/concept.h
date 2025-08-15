#pragma once

#include <concepts>
#include <ostream>
#include <span>

#include <metalchat/container.h>


namespace metalchat {


template <uint32_t N> struct tensor_layout {
    /// Sizes of a tensor.
    uint32_t sizes[N];

    /// Strides of a tensor data.
    uint32_t strides[N];

    /// Offsets of a tensor data.
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


template <std::size_t> using requires_constexpr = void;


template <typename Tensor>
concept immutable_tensor = requires(std::remove_reference_t<Tensor> const t) {
    /// The type of the elements.
    typename Tensor::value_type;
    typename Tensor::pointer_type;
    typename Tensor::container_type;
    typename Tensor::container_pointer;
    typename Tensor::iterator;
    typename Tensor::const_iterator;

    // Ensure that tensor dimension is a constexpr, so could be used in template parameters.
    { Tensor::dim() } -> std::convertible_to<std::size_t>;
    typename requires_constexpr<Tensor::dim()>;

    { t.size(std::size_t()) } -> std::same_as<std::size_t>;
    { t.sizes() } -> std::same_as<const std::span<std::size_t>>;
    { t.shape() } -> std::same_as<const std::span<std::size_t, Tensor::dim()>>;

    { t.stride(std::size_t()) } -> std::same_as<std::size_t>;
    { t.strides() } -> std::same_as<const std::span<std::size_t>>;

    { t.offset(std::size_t()) } -> std::same_as<std::size_t>;
    { t.offsets() } -> std::same_as<const std::span<std::size_t>>;
    { t.numel() } -> std::same_as<std::size_t>;

    { t.container() } -> std::same_as<typename Tensor::container_type&>;
    { t.container_ptr() } -> std::same_as<typename Tensor::container_pointer>;
    { t.layout() } -> std::same_as<tensor_layout<Tensor::dim()>>;

    { t.narrow(std::size_t(), std::size_t(), std::size_t()) } -> std::same_as<Tensor>;

    { t.data_ptr() } -> std::same_as<typename Tensor::pointer_type>;
    { t.begin() } -> std::same_as<typename Tensor::const_iterator>;
    { t.end() } -> std::same_as<typename Tensor::const_iterator>;
};


template <immutable_tensor Tensor, std::size_t N> struct change_tensor_dimensions;


template <immutable_tensor Tensor, std::size_t N>
using change_tensor_dimensions_t = typename change_tensor_dimensions<Tensor, N>::type;


template <immutable_tensor Tensor, contiguous_container Container> struct change_tensor_container;


template <immutable_tensor Tensor, contiguous_container Container>
using change_tensor_container_t = typename change_tensor_container<Tensor, Container>::type;


/// Ensures that the tensor of a given value type, so a binary operation (for example,
/// a Hadamard product) could be computed on tensors of the same value type, and never
/// on tensors of different types (meaning, no automatic type cast).
template <typename Tensor, typename T>
concept immutable_tensor_t
    = immutable_tensor<Tensor> && std::same_as<typename Tensor::value_type, T>;


template <typename Tensor, typename T>
concept immutable_scalar_t = immutable_tensor_t<Tensor, T> && Tensor::dim() == 0;


template <typename Tensor, typename T>
concept immutable_tensor1_t = immutable_tensor_t<Tensor, T> && Tensor::dim() == 1;


template <typename Tensor, typename T>
concept immutable_tensor2_t = immutable_tensor_t<Tensor, T> && Tensor::dim() == 2;


template <typename Tensor, typename T>
concept immutable_tensor3_t = immutable_tensor_t<Tensor, T> && Tensor::dim() == 3;


template <typename Tensor, typename T>
concept immutable_tensor4_t = immutable_tensor_t<Tensor, T> && Tensor::dim() == 4;


template <typename Tensor, typename T>
concept immutable_hardware_tensor_t = immutable_tensor_t<Tensor, T>
                                      && std::same_as<
                                          typename Tensor::container_type,
                                          hardware_memory_container<typename Tensor::value_type>>;


template <typename Tensor, typename T>
concept immutable_hardware_tensor4_t = immutable_tensor4_t<Tensor, T>
                                       && std::same_as<
                                           typename Tensor::container_type,
                                           hardware_memory_container<typename Tensor::value_type>>;


template <typename Tensor, typename T> struct optional_tensor : public std::false_type {};


template <typename Tensor, typename T>
struct optional_tensor<std::optional<Tensor>, T>
: public std::bool_constant<immutable_tensor_t<Tensor, T>> {};


template <typename Tensor, typename T>
concept optional_tensor_t = optional_tensor<Tensor, T>::value;


template <typename Tensor, typename T>
concept immutable_filebuf_tensor_t = immutable_tensor_t<Tensor, T>
                                     && std::same_as<
                                         typename Tensor::container_type,
                                         filebuf_memory_container<typename Tensor::value_type>>;


} // namespace metalchat
