// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/functional/transform.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/expected.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T> class conv1d {
private:
    basic_kernel _M_kernel;

    std::size_t
    compute_output_size(std::size_t input_size, std::size_t kernel_size, std::size_t padding)
    {
        return input_size + 2 * padding - (kernel_size - 1);
    }

public:
    conv1d(hardware_accelerator& accelerator)
    : _M_kernel(accelerator.load<T>("conv1d"))
    {}

    template <immutable_tensor3_t<T> Input, immutable_tensor3_t<T> Weight>
    auto
    operator()(Input input, Weight weight, std::size_t padding, std::size_t groups)
    {
        auto num_batches = input.size(0);
        auto in_channels = input.size(1);
        auto out_channels = weight.size(0);
        auto input_size = input.size(2);
        auto kernel_size = weight.size(2);
        auto output_size = compute_output_size(input_size, kernel_size, padding);
        auto alloc = _M_kernel.get_allocator();
        auto output = shared_empty<T>({num_batches, out_channels, output_size}, alloc);

        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto thread = dim3(output_size, in_channels);

        /// The thread could be scheduled either with a preference of input (sequence
        /// length) to be processed by a thread, or different output channels. Since
        /// convolution operation queries input data multiple times for the same set
        /// of input channels, we give priority to the input.
        if (output_size > max_threads) {
            thread = dim3(max_threads, 1);
        } else if (thread.numel() > max_threads) {
            thread = dim3(output_size, max_threads / output_size);
        }

        auto grid = dim3(
            thread.x * ceil_div(output_size, thread.x), thread.y * ceil_div(out_channels, thread.y),
            num_batches
        );

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(
            output, input, weight, scalar<int32_t>(padding), scalar<uint32_t>(groups)
        );

        return future_tensor(output, std::move(task_future));
    }

    template <immutable_tensor2_t<T> Input, immutable_tensor3_t<T> Weight>
    auto
    operator()(Input input, Weight weight, std::size_t padding, std::size_t groups)
    {
        auto output = operator()(input.expand_dims(0), weight, padding, groups);
        return output[0];
    }
};


} // namespace kernel
} // namespace metalchat
