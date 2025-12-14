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


template <typename T, std::size_t BlockSize = 16> class rmsnorm {
private:
    basic_kernel _M_kernel;

public:
    rmsnorm(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("rmsnorm", BlockSize))
    {}

    template <immutable_tensor_t<T> Input, immutable_tensor1_t<T> Weight>
    auto
    operator()(Input input, Weight weight, const float eps = 1e-5)
    {
        auto dim_size = input.sizes().back();
        auto expected_weight = expected_tensor(weight).same_dim(0, /*expect=*/dim_size).value();

        auto input_view = flatten<2>(input);
        auto output_view = shared_empty_like<T>(input_view, _M_kernel.get_allocator());

        auto [grid, thread] = make_kernel_grid_1d(input, BlockSize);

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future =
            task.bind_front(output_view, input_view, expected_weight, scalar<float>(eps));

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input.shape());
    }
};


} // namespace kernel
} // namespace metalchat
