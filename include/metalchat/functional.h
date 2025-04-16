#pragma once

#include <cstring>
#include <format>
#include <functional>
#include <ranges>
#include <span>

#include <metalchat/kernel/bmm.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/kernel/mul.h>
#include <metalchat/kernel/silu.h>
#include <metalchat/kernel/softmax.h>
#include <metalchat/kernel/sum.h>
#include <metalchat/tensor_future.h>


namespace metalchat {
namespace fn {


template <immutable_tensor Tensor1, immutable_tensor Tensor2>
auto
matmul(Tensor1 t1, Tensor2 t2, device& gpu)
{
    bmm<typename Tensor1::value_type> op(gpu);
    return op(t1, t2);
}


template <immutable_tensor Tensor, typename T>
auto
mul(Tensor t, const T multiplier, device& gpu)
{
    scalar_mul<typename Tensor::value_type> op(gpu);
    return op(t, multiplier);
}


template <immutable_tensor Tensor1, immutable_tensor Tensor2>
auto
hadamard(Tensor1 t1, Tensor2 t2, device& gpu)
{
    metalchat::hadamard<typename Tensor1::value_type> op(gpu);
    return op(t1, t2);
}


template <immutable_tensor Tensor1, immutable_tensor Tensor2>
auto
sum(Tensor1 t1, Tensor2 t2, device& gpu)
{
    metalchat::sum<typename Tensor1::value_type> op(gpu);
    return op(t1, t2);
}


template <immutable_tensor Tensor1, immutable_tensor Tensor2>
auto
sum2(Tensor1 t1, Tensor2 t2, device& gpu)
{
    metalchat::sum2<typename Tensor1::value_type> op(gpu);
    return op(t1, t2);
}


template <immutable_tensor Tensor>
auto
softmax(Tensor t, device& gpu)
{
    metalchat::softmax<typename Tensor::value_type> op(gpu);
    return op(t);
}


template <immutable_tensor Tensor>
auto
silu(Tensor t, device& gpu)
{
    metalchat::silu<typename Tensor::value_type> op(gpu);
    return op(t);
}


} // namespace fn


template <typename T, forward_tensor_iterator_t<T> ForwardIt>
auto
concatenate(ForwardIt begin, ForwardIt end, std::size_t dim, device& device)
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
        // assert((*first).is_contiguous());

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

    auto output = future_tensor(empty<T>(std::move(size0), device));
    std::size_t offset = 0;

    auto copy_kernel = cpy<T>(device);

    for (auto first = begin; first != end; ++first) {
        const auto& input = (*first);
        auto n = input.size(dim);
        auto target = output.narrow(dim, offset, n);

        output = future_tensor(output, copy_kernel(input, target));
        offset += n;
    }

    return output;
}


template <immutable_tensor Tensor>
auto
concatenate(const std::initializer_list<Tensor> tensors, std::size_t dim, device& device)
{
    using T = Tensor::value_type;
    return concatenate<T>(tensors.begin(), tensors.end(), dim, device);
}

template <immutable_tensor Tensor>
auto
repeat_interleave(Tensor t, std::size_t repeats, std::size_t dim, device& device)
{
    using T = Tensor::value_type;

    auto expanded_tensor = t.expand_dims(dim + 1);
    auto rep_tensor = std::views::repeat(expanded_tensor, repeats);
    auto output = concatenate<T>(rep_tensor.begin(), rep_tensor.end(), dim + 1, device);
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
