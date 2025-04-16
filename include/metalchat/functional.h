#pragma once

#include <concepts>
#include <cstring>
#include <format>
#include <functional>
#include <iterator>
#include <ranges>
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

    // Ensure that sizes of all concatenated tensors are the same.
    std::size_t size0[tensor_type::dim()];
    std::copy((*begin).sizes().begin(), (*begin).sizes().end(), size0);
    size0[dim] = 0;

    for (auto first = begin; first != end; ++first) {
        assert((*first).is_contiguous());

        auto sizes = (*first).sizes();
        for (auto i = 0; i < tensor_type::dim(); i++) {
            if (i != dim && sizes[i] != size0[i]) {
                throw std::invalid_argument("unable to concatenate tensor of various shapes");
            }
            if (i == dim) {
                size0[dim] += sizes[dim];
            }
        }
    }

    auto output = empty<value_type>(std::move(size0));
    auto output_ptr = output.data_ptr();
    std::size_t offset = 0;

    for (auto first = begin; first != end; ++first) {
        const auto& input = (*first);
        auto n = input.size(dim);
        auto target = output.narrow(dim, offset, n);

        std::copy(input.begin(), input.end(), target.begin());
        offset += n;
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
    using tensor_const_ref = std::reference_wrapper<const tensor_type>;

    auto reference_iterator
        = std::views::transform(tensors, [](tensor_const_ref ref) -> const tensor_type& {
        return ref.get();
    });

    return concatenate(reference_iterator.begin(), reference_iterator.end(), dim);
}


template <typename T, std::size_t N, ContiguousContainer Container>
auto
repeat_interleave(tensor<T, N, Container>&& t, std::size_t repeats, std::size_t dim)
{
    auto exp_tensor = tensor(t.expand_dims(dim + 1));
    auto rep_tensor = std::views::repeat(std::move(exp_tensor), repeats);
    auto output = concatenate(rep_tensor.begin(), rep_tensor.end(), dim + 1);
    return output;
}


template <typename T, ContiguousContainer Container>
void
triu(tensor<T, 2, Container>& t)
{
    for (auto i = 0; i < t.size(0); i++) {
        for (auto j = 0; j <= i; j++) {
            t[i][j] = T(0);
        }
    }
}


} // namespace metalchat
