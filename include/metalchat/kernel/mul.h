// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/expected.h>
#include <metalchat/tensor/format.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T> class hadamard {
private:
    binary_kernel_wrapper<T> _M_kernel;

public:
    hadamard(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("hadamard"))
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        return _M_kernel(input1, input2);
    }
};


template <typename T, typename I1, typename I2, std::size_t BlockSize = 16>
class hadamard_broadcast {
private:
    basic_kernel _M_kernel;

public:
    hadamard_broadcast(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T, I1, I2>("hadamard_broadcast", BlockSize))
    {}

    template <immutable_tensor2_t<I1> Input1, immutable_tensor2_t<I2> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        auto expected_input1 = expected_tensor(input1).same_first_dim(input2).value();
        auto expected_input2 = expected_tensor(input2).same_dim(1, /*expect=*/1).value();

        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto num_rows = ceil_div(input1.size(0), BlockSize);
        auto num_dims = ceil_div(input1.size(1), BlockSize);

        auto thread = dim3(ceil_div(max_threads, num_dims), num_dims);
        auto grid = dim3(ceil_div(num_rows, thread.x) * thread.x, thread.y);

        auto output = shared_empty_like<T>(input1, _M_kernel.get_allocator());

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output, input1, input2);

        return future_tensor(output, std::move(task_future));
    }

    template <immutable_tensor3_t<I1> Input1, immutable_tensor3_t<I2> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        auto expected_input1 = expected_tensor(input1).same_dim(input2, 1, 1).value();

        auto input1_view = flatten<2>(expected_input1);
        auto input2_view = flatten<2>(input2);

        auto output = operator()(input1_view, input2_view);
        return output.view(expected_input1.shape());
    }
};


template <typename T> class scalar_mul {
private:
    binary_kernel_wrapper<T> _M_kernel;

public:
    scalar_mul(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("scalar_mul"))
    {}

    template <immutable_tensor_t<T> Input, immutable_scalar_t<T> Multiplier>
    auto
    operator()(Input input, Multiplier multiplier) requires(Input::dim() > 1)
    {
        return _M_kernel(input, multiplier);
    }

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input, const T multiplier) requires(Input::dim() > 1)
    {
        return _M_kernel(input, multiplier);
    }
};


} // namespace kernel
} // namespace metalchat
