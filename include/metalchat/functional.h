// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <cstring>
#include <format>
#include <functional>
#include <ranges>
#include <span>

#include <metalchat/kernel/arithmetic.h>
#include <metalchat/kernel/bmm.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/kernel/cumsum.h>
#include <metalchat/kernel/logical.h>
#include <metalchat/kernel/mul.h>
#include <metalchat/kernel/multinomial.h>
#include <metalchat/kernel/roll.h>
#include <metalchat/kernel/silu.h>
#include <metalchat/kernel/softmax.h>
#include <metalchat/kernel/sort.h>
#include <metalchat/tensor/future.h>


namespace metalchat {


template <immutable_tensor Tensor1, immutable_tensor Tensor2, std::size_t BlockSize = 8>
auto
matmul(Tensor1 t1, Tensor2 t2, hardware_accelerator& gpu)
{
    kernel::bmm<typename Tensor1::value_type, BlockSize> op(gpu);
    return op(t1, t2);
}


template <immutable_tensor Tensor>
auto
mul(Tensor t, const typename Tensor::value_type multiplier, hardware_accelerator& gpu)
{
    kernel::scalar_mul<typename Tensor::value_type> op(gpu);
    return op(t, multiplier);
}


template <immutable_tensor Tensor1, immutable_tensor Tensor2>
auto
hadamard(Tensor1 t1, Tensor2 t2, hardware_accelerator& gpu)
{
    kernel::hadamard<typename Tensor1::value_type> op(gpu);
    return op(t1, t2);
}


template <
    typename T,
    immutable_tensor Tensor1,
    immutable_tensor Tensor2,
    std::size_t BlockSize = 16>
auto
hadamard_broadcast(Tensor1 t1, Tensor2 t2, hardware_accelerator& gpu)
{
    using input1_type = Tensor1::value_type;
    using input2_type = Tensor2::value_type;
    kernel::hadamard_broadcast<T, input1_type, input2_type> op(gpu);
    return op(t1, t2);
}


template <immutable_tensor Tensor1, immutable_tensor Tensor2>
auto
add(Tensor1 t1, Tensor2 t2, hardware_accelerator& gpu)
{
    kernel::add<typename Tensor1::value_type> op(gpu);
    return op(t1, t2);
}


template <immutable_tensor Tensor1, immutable_tensor Tensor2, std::size_t BlockSize = 8>
auto
add2(Tensor1 t1, Tensor2 t2, hardware_accelerator& gpu)
{
    kernel::add2<typename Tensor1::value_type, BlockSize> op(gpu);
    return op(t1, t2);
}


template <immutable_tensor Tensor, std::size_t BlockSize = 16>
auto
softmax(Tensor t, hardware_accelerator& gpu)
{
    kernel::softmax<typename Tensor::value_type, BlockSize> op(gpu);
    return op(t);
}


template <immutable_tensor Tensor>
auto
silu(Tensor t, hardware_accelerator& gpu)
{
    kernel::silu<typename Tensor::value_type> op(gpu);
    return op(t);
}


template <immutable_tensor Tensor>
auto
sort(Tensor t, hardware_accelerator& gpu)
{
    kernel::sort<typename Tensor::value_type> op(gpu);
    return op(t);
}


template <immutable_tensor Tensor>
auto
roll(Tensor t, int32_t shift, std::size_t dim, hardware_accelerator& gpu)
{
    kernel::roll<typename Tensor::value_type> op(gpu);
    return op(t, shift, dim);
}


template <immutable_tensor Input, immutable_tensor Output>
auto
roll(Input input, Output output, int32_t shift, std::size_t dim, hardware_accelerator& gpu)
{
    kernel::roll<typename Input::value_type> op(gpu);
    return op(input, output, shift, dim);
}

template <immutable_tensor Tensor, std::size_t BlockSize = 16>
auto
cumsum(Tensor t, hardware_accelerator& gpu)
{
    kernel::cumsum<typename Tensor::value_type, BlockSize> op(gpu);
    return op(t);
}


template <typename T, immutable_tensor_t<T> Tensor, immutable_tensor_t<bool> Mask>
auto
scatter(Tensor t, Mask m, T value, hardware_accelerator& gpu)
{
    kernel::scatter<T> op(gpu);
    return op(t, m, value);
}


template <immutable_tensor Tensor, immutable_tensor_t<int32_t> Index>
auto
gather(Tensor t, Index index, hardware_accelerator& gpu)
{
    kernel::gather<typename Tensor::value_type> op(gpu);
    return op(t, index);
}


template <immutable_tensor Tensor>
auto
clone(Tensor t, hardware_accelerator& gpu)
{
    kernel::clone<typename Tensor::value_type> op(gpu);
    return op(t);
}


template <immutable_tensor Tensor1, immutable_tensor Tensor2>
requires(std::same_as<typename Tensor1::value_type, typename Tensor2::value_type>)
auto
sub(Tensor1 t1, Tensor2 t2, hardware_accelerator& gpu)
{
    kernel::sub<typename Tensor1::value_type> op(gpu);
    return op(t1, t2);
}


template <typename T, immutable_tensor_t<T> Tensor>
auto
gt(Tensor t, T value, hardware_accelerator& gpu)
{
    kernel::gt<T> op(gpu);
    return op(t, value);
}


template <typename T, immutable_tensor_t<T> Tensor>
auto
le(Tensor t, T value, hardware_accelerator& gpu)
{
    kernel::le<T> op(gpu);
    return op(t, value);
}


template <immutable_tensor Tensor>
auto
multinomial(Tensor t, std::size_t sample_size, hardware_accelerator& gpu)
{
    kernel::multinomial<typename Tensor::value_type> op(gpu);
    return op(t, sample_size);
}


template <typename T, immutable_tensor2_t<T> Tensor, std::size_t BlockSize = 128>
auto
top_p(Tensor logits, T temperature, T p, hardware_accelerator& gpu)
{
    logits = mul(logits, T(1) / temperature, gpu);
    auto probs = softmax<Tensor, BlockSize>(logits, gpu);

    auto [probs_sort, probs_idx] = sort(probs, gpu);
    auto probs_sum = cumsum<Tensor, BlockSize>(probs_sort, gpu);
    auto probs_diff = sub(probs_sum, probs_sort, gpu);

    auto mask = gt(probs_diff, p, gpu);
    probs_sort = scatter(probs_sort, mask, T(0), gpu);

    auto next_token = multinomial(probs_sort, /*sample_size=*/1, gpu);
    return gather(probs_idx, next_token, gpu);
}


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

    auto clone = kernel::clone<T>(gpu);

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


template <typename T, contiguous_container Container>
void
triu(tensor<T, 2, Container>& t)
{
    for (std::size_t i = 0; i < t.size(0); i++) {
        for (std::size_t j = 0; j <= i && j < t.size(1); j++) {
            t[i][j] = T(0);
        }
    }
}

template <typename T, contiguous_container Container>
void
triu(tensor<T, 2, Container>&& t)
{
    triu(t);
}


} // namespace metalchat
