#pragma once

#include <concepts>
#include <cstring>
#include <format>
#include <functional>
#include <iterator>
#include <span>

#include <metalchat/container.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename Tensor>
concept is_tensor = requires(Tensor t) {
    typename Tensor::value_type;
    typename Tensor::pointer_type;
    typename Tensor::container_type;

    { Tensor::dim() } -> std::convertible_to<std::size_t>;
    { t.sizes() } -> std::same_as<const std::span<std::size_t>>;
    { t.numel() } -> std::same_as<std::size_t>;
    { t.data_ptr() } -> std::same_as<typename Tensor::pointer_type>;
};


template <typename It>
concept TensorIterator = std::forward_iterator<It> && requires(It it) {
    typename std::iterator_traits<It>::value_type;
    requires is_tensor<typename std::iterator_traits<It>::value_type>;
};


template <TensorIterator ForwardIt>
auto
concatenate(ForwardIt begin, ForwardIt end, std::size_t dim)
{
    using tensor_type = std::iterator_traits<ForwardIt>::value_type;
    using value_type = tensor_type::value_type;

    if (dim > tensor_type::dim()) {
        throw std::invalid_argument(std::format(
            "invalid dim ({}) passed to concatenate {}-dimensional tensors", dim, tensor_type::dim()
        ));
    }
    if (begin == end) {
        throw std::invalid_argument("expected non-empty list of tensors");
    }

    // Ensure that sizes of all concatenated tensors are the same (TODO: with exception
    // to a concatenating dimension).
    auto size0 = (*begin).sizes();
    std::size_t num_tensors = 0;

    for (auto first = begin; first != end; ++first, ++num_tensors) {
        assert((*first).is_contiguous());

        auto sizes = (*first).sizes();
        for (auto i = 0; i < size0.size(); i++) {
            if (sizes[i] != size0[i]) {
                throw std::invalid_argument("unable to concatenate tensor of various shapes");
            }
        }
    }

    std::size_t sizes[tensor_type::dim() + 1];
    sizes[dim] = num_tensors;

    for (auto i = 0; i < dim; i++) {
        sizes[i] = size0[i];
    }
    for (auto i = dim; i < tensor_type::dim(); i++) {
        sizes[i + 1] = size0[i];
    }

    auto output = empty<value_type>(std::move(sizes));
    std::size_t offset = 0;

    for (auto first = begin; first != end; ++first) {
        auto numel = (*first).numel();
        std::memcpy(output.data_ptr() + offset, (*first).data_ptr(), numel * sizeof(value_type));
        offset += numel;
    }

    return output;
}


template <typename T, std::size_t N, ContiguousContainer Container>
auto
concatenate(
    std::initializer_list<std::reference_wrapper<const tensor<T, N, Container>>> tensors,
    std::size_t dim
)
{
    using tensor_type = tensor<T, N, Container>;

    auto reference_iterator = tensors
                              | std::views::transform(
                                  [](std::reference_wrapper<const tensor_type> ref
                                  ) -> const tensor_type& { return ref.get(); }
                              );

    return concatenate(reference_iterator.begin(), reference_iterator.end(), dim);
}


} // namespace metalchat
