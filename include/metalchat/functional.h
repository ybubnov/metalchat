#pragma once

#include <cstring>
#include <format>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>

#include <metalchat/container.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_concept.h>
#include <metalchat/tensor_future.h>
#include <metalchat/tensor_shared.h>


namespace metalchat {


template <typename It>
concept forward_tensor_iterator = std::forward_iterator<It> && requires(It it) {
    typename std::iterator_traits<It>::value_type;
    requires immutable_tensor<typename std::iterator_traits<It>::value_type>;
};


template <forward_tensor_iterator ForwardIt>
auto
concatenate(ForwardIt begin, ForwardIt end, std::size_t dim, device& device)
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

    auto output = shared_tensor(empty<value_type>(std::move(size0), device));
    std::size_t offset = 0;

    std::vector<std::shared_ptr<awaitable>> futures;
    auto copy_kernel = cpy<value_type>(device);

    for (auto first = begin; first != end; ++first) {
        const auto& input = (*first);
        auto n = input.size(dim);
        auto target = output.narrow(dim, offset, n);

        futures.push_back(make_shared(copy_kernel(input, target)));
        offset += n;
    }

    wait_all(futures);
    return output;
}


template <typename T, std::size_t N, ContiguousContainer Container>
auto
concatenate(
    const std::initializer_list<shared_tensor<T, N, Container>> tensors,
    std::size_t dim,
    device& device
)
{
    return concatenate(tensors.begin(), tensors.end(), dim, device);
}


template <typename T, std::size_t N, ContiguousContainer Container>
auto
repeat_interleave(
    shared_tensor<T, N, Container> t, std::size_t repeats, std::size_t dim, device& device
)
{
    auto expanded_tensor = t.expand_dims(dim + 1);
    auto rep_tensor = std::views::repeat(expanded_tensor, repeats);
    auto output = concatenate(rep_tensor.begin(), rep_tensor.end(), dim + 1, device);
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
