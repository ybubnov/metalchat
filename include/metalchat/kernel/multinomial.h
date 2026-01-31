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


/// Draw samples from a multinomial distribution.
///
/// Input of this method should be a cumulative distribution function of a multinomial
/// distribution. Values in each row of the input should be a between 0.0 to 1.0, since
/// implementation uses a uniform value generator to sample from CDF.
///
/// The kernel expects input probabilities to be in reverse order.
template <typename T> class multinomial {
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
        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto [grid, thread] = make_kernel_grid_2d(num_rows, dim_size, max_threads);

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
