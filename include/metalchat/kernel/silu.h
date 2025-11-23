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


template <typename T, std::size_t BlockSize = 16> class silu {
private:
    basic_kernel _M_kernel;

public:
    silu(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("silu"))
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        auto input_view = flatten<2>(input);
        auto output_view = shared_empty_like<T>(input_view, _M_kernel.get_allocator());

        auto [grid, thread] = make_kernel_grid_2d(input, BlockSize);

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input_view);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input.shape());
    }
};


} // namespace kernel
} // namespace metalchat
