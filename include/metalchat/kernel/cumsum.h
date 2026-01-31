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


} // namespace kernel
} // namespace metalchat
