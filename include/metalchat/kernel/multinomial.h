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


template <typename T, std::size_t BlockSize = 32> class multinomial {
private:
    basic_kernel _M_kernel;

    std::random_device _M_random_device;
    std::mt19937 _M_generator;
    std::uniform_int_distribution<uint64_t> _M_seed;

public:
    multinomial(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("multinomial")),
      _M_random_device(),
      _M_generator(_M_random_device()),
      _M_seed(0, std::numeric_limits<uint64_t>::max())
    {}

    template <immutable_tensor2_t<T> Input>
    auto
    operator()(Input input, std::size_t sample_size)
    {
        auto num_rows = input.size(0);
        auto dim_size = sample_size;

        auto thread_size = ceil_div(dim_size, BlockSize);
        auto thread = dim3(thread_size);
        auto grid = dim3(thread_size * num_rows, BlockSize);

        auto init_state = _M_seed(_M_generator);
        auto init_seq = _M_seed(_M_generator);

        auto alloc = _M_kernel.get_allocator();
        auto output = shared_empty<int32_t>({num_rows, sample_size}, alloc);

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output, input, scalar(init_state), scalar(init_seq));

        return future_tensor(output, std::move(task_future));
    }
};


} // namespace kernel
} // namespace metalchat
