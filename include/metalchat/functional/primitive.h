// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/kernel/arithmetic.h>
#include <metalchat/kernel/bmm.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/kernel/logical.h>
#include <metalchat/kernel/mul.h>
#include <metalchat/kernel/multinomial.h>
#include <metalchat/kernel/roll.h>
#include <metalchat/kernel/silu.h>
#include <metalchat/kernel/softmax.h>
#include <metalchat/kernel/sort.h>
#include <metalchat/kernel/sum.h>
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


template <immutable_tensor Tensor>
auto
softmax(Tensor t, hardware_accelerator& gpu)
{
    kernel::softmax<typename Tensor::value_type> op(gpu);
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

template <immutable_tensor Tensor>
auto
cumsum(Tensor t, hardware_accelerator& gpu)
{
    kernel::cumsum<typename Tensor::value_type> op(gpu);
    return op(t);
}


template <immutable_tensor Tensor>
auto
sum(Tensor t, hardware_accelerator& gpu)
{
    kernel::sum<typename Tensor::value_type> op(gpu);
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


template <immutable_tensor Tensor1, immutable_tensor Tensor2>
requires(std::same_as<typename Tensor1::value_type, typename Tensor2::value_type>)
auto
div(Tensor1 input1, Tensor2 input2, hardware_accelerator& gpu)
{
    kernel::div<typename Tensor1::value_type> op(gpu);
    return op(input1, input2);
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
