// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/expected.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T> class bmm {
private:
    basic_kernel _M_kernel8;
    basic_kernel _M_kernel16;
    basic_kernel _M_kernel32;

public:
    bmm(hardware_accelerator& gpu)
    : _M_kernel8(gpu.load<T>("bmm", 8)),
      _M_kernel16(gpu.load<T>("bmm", 16)),
      _M_kernel32(gpu.load<T>("bmm", 32))
    {}

    template <immutable_tensor3_t<T> Input, immutable_tensor3_t<T> Weight>
    auto
    operator()(Input input, Weight weight)
    {
        auto num_batches = input.size(0);
        auto input_size1 = input.size(1);
        auto weight_size2 = weight.size(2);

        std::size_t block_size = 8;
        auto* kernel = &_M_kernel8;
        if (weight_size2 > 8) {
            kernel = &_M_kernel16;
            block_size = 16;
        }
        if (weight_size2 > 16) {
            kernel = &_M_kernel32;
            block_size = 32;
        }

        // Batched matmul does not support broadcasting operations, therefore throw an
        // exception, when the number of batches for input tensors are different.
        auto expected_input =
            expected_tensor(input).same_dim(weight, 0).same_dim(weight, 2, 1).value();

        auto alloc = kernel->get_allocator();
        auto output = shared_empty<T>({num_batches, input_size1, weight_size2}, alloc);

        auto grid = dim3(
            ceil_div(input_size1, block_size) * block_size / 2,
            ceil_div(weight_size2, block_size) * block_size, num_batches
        );
        auto thread = dim3(block_size / 2, block_size);

        auto task = kernel_task(*kernel, grid, thread);
        auto task_future = task.bind_front(output, expected_input, weight);

        // A(MxK) @ B(KxN) -> C(MxN)
        return future_tensor(output, std::move(task_future));
    }

    template <immutable_tensor3_t<T> Input, immutable_tensor2_t<T> Weight>
    auto
    operator()(Input input, Weight weight)
    {
        // TODO: does it make sense to call repeat_interleave for the number of batches > 1?
        return operator()(input, weight.expand_dims(0));
    }

    template <immutable_tensor2_t<T> Input, immutable_tensor2_t<T> Weight>
    auto
    operator()(Input input, Weight weight)
    {
        auto output = operator()(input.expand_dims(0), weight.expand_dims(0));

        int input_size0 = input.size(0);
        int weight_size1 = weight.size(1);
        return output.view({input_size0, weight_size1});
    }

    template <immutable_tensor_t<T> Input, immutable_tensor_t<T> Weight>
    auto
    operator()(Input input, Weight weight)
        requires(Input::dim() == Weight::dim() && Input::dim() > 3)
    {
        constexpr std::size_t N = Input::dim();

        int output_sizes[N];
        std::copy(input.sizes().begin(), input.sizes().end(), output_sizes);
        output_sizes[N - 2] = input.size(N - 2);
        output_sizes[N - 1] = weight.size(N - 1);

        auto output = operator()(flatten<3>(input), flatten<3>(weight));
        return output.view(std::move(output_sizes));
    }
};


} // namespace kernel
} // namespace metalchat
