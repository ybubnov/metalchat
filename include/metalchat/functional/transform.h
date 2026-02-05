// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <cstring>
#include <format>
#include <functional>
#include <ranges>
#include <span>

#include <metalchat/kernel/copy.h>
#include <metalchat/tensor/future.h>


namespace metalchat {


template <typename T, forward_tensor_iterator_t<T> ForwardIt>
auto
concatenate(ForwardIt begin, ForwardIt end, std::size_t dim, hardware_accelerator& gpu)
{
    using tensor_type = std::iterator_traits<ForwardIt>::value_type;

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
        auto sizes = (*first).sizes();

        for (std::size_t i = 0; i < tensor_type::dim(); i++) {
            if (i != dim && sizes[i] != size0[i]) {
                throw std::invalid_argument("unable to concatenate tensor of various shapes");
            }
            if (i == dim) {
                size0[dim] += sizes[dim];
            }
        }
    }

    auto output = future_tensor(empty<T>(std::move(size0), gpu.get_allocator()));
    std::size_t offset = 0;

    kernel::clone<T> clone(gpu);

    for (auto first = begin; first != end; ++first) {
        const auto& input = (*first);
        auto n = input.size(dim);
        auto target = output.narrow(dim, offset, n);

        output = future_tensor(output, clone(input, target));
        offset += n;
    }

    return output;
}


template <immutable_tensor Tensor>
auto
concatenate(const std::initializer_list<Tensor> tensors, std::size_t dim, hardware_accelerator& gpu)
{
    using T = Tensor::value_type;
    return concatenate<T>(tensors.begin(), tensors.end(), dim, gpu);
}


template <immutable_tensor Tensor>
auto
repeat_interleave(Tensor t, std::size_t repeats, std::size_t dim, hardware_accelerator& gpu)
{
    using T = Tensor::value_type;

    auto expanded_tensor = t.expand_dims(dim + 1);
    auto rep_tensor = std::views::repeat(expanded_tensor, repeats);
    auto output = concatenate<T>(rep_tensor.begin(), rep_tensor.end(), dim + 1, gpu);
    return output;
}


} // namespace metalchat
