// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T> class cumsum {
private:
    // The kernel is loaded dynamically, but this one is necessary to
    // query maximum threads allowed to schedule within a single threadgroup.
    basic_kernel _M_kernel;

public:
    cumsum(hardware_accelerator& accelerator)
    : _M_kernel(accelerator.load<T>("cumsum", 2))
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        auto dim_size = input.sizes().back();
        auto num_rows = input.numel() / dim_size;

        // Cumulative sum kernel uses stack-allocated memory, and size of this memory
        // is required to be a constant expression. It means we cannot pass it as a parameter
        // to the kernel. Here we load a necessary kernel dynamically based on block size
        // ceiled to the nearest power of 2.
        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto block_size = ceil_pow2(ceil_div(dim_size, max_threads));
        block_size = std::max(std::size_t(2), block_size);
        auto thread_size = ceil_div(dim_size, block_size);

        auto thread = dim3(thread_size);
        auto grid = dim3(thread_size * num_rows);

        auto alloc = _M_kernel.get_allocator();
        auto input_view = flatten<2>(input);
        auto output_view = shared_empty_like<T>(input_view, alloc);

        auto& accelerator = _M_kernel.get_accelerator();
        auto kernel = accelerator.template load<T>("cumsum", block_size);
        auto task = kernel_task(kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input_view);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input.shape());
    }
};


/// Return the sum of each row of the `input` tensor in the last dimension.
///
/// ```c++
/// auto input = tensor<float>({{1.0, 2.0, 3.0}, {3.0, 4.0, 5.0}});
///
/// auto accelerator = hardware_accelerator();
/// auto sum = kernel::sum<float>(accelerator);
///
/// auto output = sum(input);
/// // out:
/// // [6.0, 12.0], sizes=(2)
/// ```
template <typename T> class sum {
private:
    basic_kernel _M_kernel;

public:
    /// The kernel constructor.
    sum(hardware_accelerator& accelerator)
    : _M_kernel(accelerator.load<T>("sum"))
    {}

    /// Invokes the kernel.
    ///
    /// \param input the input tensor.
    /// \returns a \ref future_tensor with the result.
    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input) requires(Input::dim() > 1)
    {
        auto input_sizes = input.sizes();
        auto dim_size = input_sizes.back();
        auto num_rows = input.numel() / dim_size;

        auto input_view = flatten<2>(input);
        auto output_view = shared_empty<T>({num_rows}, _M_kernel.get_allocator());

        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto block_size = ceil_div(dim_size, max_threads);
        auto thread_size = ceil_div(dim_size, block_size);

        auto thread = dim3(thread_size);
        auto grid = dim3(thread_size * num_rows);

        auto block_tensor = scalar<uint32_t>(block_size);
        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input_view, block_tensor);

        auto output = future_tensor(output_view, std::move(task_future));

        constexpr auto output_dim = Input::dim() - 1;
        std::size_t output_sizes[output_dim];
        std::copy_n(input_sizes.begin(), output_dim, output_sizes);

        return output.view(std::span(output_sizes));
    }
};


} // namespace kernel
} // namespace metalchat
