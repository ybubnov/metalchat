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


template <typename T> class rmsnorm {
private:
    basic_kernel _M_kernel;

public:
    rmsnorm(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("rmsnorm"))
    {}

    template <immutable_tensor_t<T> Input, immutable_tensor1_t<T> Weight>
    auto
    operator()(Input input, Weight weight, const float eps = 1e-5)
    {
        auto dim_size = input.sizes().back();
        auto num_rows = input.numel() / dim_size;

        auto expected_weight = expected_tensor(weight).same_dim(0, /*expect=*/dim_size).value();

        auto input_view = flatten<2>(input);
        auto output_view = shared_empty_like<T>(input_view, _M_kernel.get_allocator());

        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto block_size = ceil_div(dim_size, max_threads);
        auto thread_size = ceil_div(dim_size, block_size);

        auto thread = dim3(thread_size);
        auto grid = dim3(thread_size * num_rows);

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(
            output_view, input_view, expected_weight, scalar<float>(eps),
            scalar<uint32_t>(block_size)
        );

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input.shape());
    }
};


} // namespace kernel
} // namespace metalchat
