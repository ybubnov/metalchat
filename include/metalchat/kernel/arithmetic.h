// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <format>

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 32> class add {
private:
    binary_kernel_wrapper<T, BlockSize> _M_kernel;

public:
    add(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T, BlockSize>("add"))
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        return _M_kernel(input1, input2);
    }
};


template <typename T, std::size_t BlockSize = 8> class add2 {
private:
    basic_kernel _M_kernel;

public:
    add2(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T, BlockSize>("add2"))
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor2_t<T> Input2>
    requires(Input1::dim() >= 2)
    auto
    operator()(Input1 input1, Input2 input2)
    {
        constexpr auto M = Input1::dim();

        auto data_size = input1.numel();
        auto dim0_size = input2.size(0);
        auto dim1_size = input2.size(1);
        auto num_rows = data_size / (dim0_size * dim1_size);

        if (dim0_size != input1.size(M - 2) || dim1_size != input1.size(M - 1)) {
            throw std::invalid_argument(std::format(
                "add2: last dimensions should be the same for both tensors {}x{} != {}x{}",
                input1.size(M - 2), input1.size(M - 1), dim0_size, dim1_size
            ));
        }

        auto input1_view = input1.view({-1, int(dim0_size), int(dim1_size)});
        auto output_view = shared_empty_like<T>(input1_view, _M_kernel.get_allocator());

        auto thread_size_x = ceil_div(dim0_size, BlockSize);
        auto thread_size_z = ceil_div(dim1_size, BlockSize);
        auto thread = dim3(thread_size_x, 1, thread_size_z);
        auto grid = dim3(thread_size_x * num_rows, BlockSize, thread_size_z);

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input1_view, input2);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input1.shape());
    }
};


template <typename T, std::size_t BlockSize = 32> class sub {
private:
    binary_kernel_wrapper<T, BlockSize> _M_kernel;

public:
    sub(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T, BlockSize>("sub"))
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        return _M_kernel(input1, input2);
    }
};

} // namespace kernel
} // namespace metalchat
