// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <format>

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/functional/transform.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/expected.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T> class add {
private:
    binary_kernel_wrapper<T> _M_kernel;

public:
    add(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("add"))
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
        requires(Input1::dim() > 1 && Input1::dim() == Input2::dim())
    {
        return _M_kernel(input1, input2);
    }
};


template <typename T> class add_broadcast {
private:
    basic_kernel _M_kernel;

public:
    add_broadcast(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("add_broadcast"))
    {}

    template <immutable_tensor3_t<T> Input1, immutable_tensor2_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
    {
        constexpr auto M = Input1::dim();

        auto expected_input1 =
            expected_tensor(input1).same_dim(input2, M - 2, 0).same_dim(input2, M - 1, 1).value();

        auto dim0_size = input2.size(0);
        auto dim1_size = input2.size(1);
        auto num_batches = input1.sizes().front();

        auto alloc = _M_kernel.get_allocator();
        auto output = shared_empty_like<T>(input1, alloc);

        auto max_threads = _M_kernel.max_threads_per_threadgroup();

        auto q = double(dim0_size) / double(dim0_size + dim1_size);
        auto threads_ratio = double(max_threads) * q;

        constexpr std::size_t one = 1;
        auto max_threads_x = std::max(one, std::size_t(threads_ratio));
        auto max_threads_y = std::max(one, std::size_t(std::floor(max_threads / max_threads_x)));


        auto block_size_x = ceil_div(dim0_size, max_threads_x);
        auto block_size_y = ceil_div(dim1_size, max_threads_y);

        auto thread_size_x = ceil_div(dim0_size, block_size_x);
        auto thread_size_y = ceil_div(dim1_size, block_size_y);

        auto thread = dim3(thread_size_x, thread_size_y);

        auto grid_size_x = thread_size_x * ceil_div(dim0_size, thread_size_x);
        auto grid_size_y = thread_size_y * ceil_div(dim1_size, thread_size_y);
        auto grid = dim3(grid_size_x, grid_size_y, num_batches);

        auto block_size = scalar<uint32_t>(block_size_x);
        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output, input1, input2, block_size);

        return future_tensor(output, std::move(task_future));
    }

    template <immutable_tensor_t<T> Input1, immutable_tensor2_t<T> Input2>
    requires(Input1::dim() > 3)
    auto
    operator()(Input1 input1, Input2 input2)
    {
        auto output = operator()(flatten<3>(input1), input2);
        return output.view(input1.shape());
    }
};


template <typename T> class sub {
private:
    binary_kernel_wrapper<T> _M_kernel;

public:
    sub(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("sub"))
    {}

    template <immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
        requires(Input1::dim() > 1 && Input1::dim() == Input2::dim())
    {
        return _M_kernel(input1, input2);
    }
};


/// Divides each element of the `input1` by corresponding element of `input2`.
///
/// \note The kernel performs true division. The kernel does not support type promotion.
///
/// ```cpp
/// auto input1 = tensor<float>({{3.0, 6.0, 9.0}});
/// auto input2 = tensor<float>({{1.0, 2.0, 3.0}});
///
/// auto accelerator = hardware_accelerator();
/// auto div = kernel::div<float>(accelerator);
///
/// auto output = div(input1, input2);
/// std::cout << output.get() << std::endl;
/// // out:
/// // [[3.0, 3.0, 3.0]], sizes=(1, 3)
/// ```
template <typename T> class div {
private:
    binary_kernel_wrapper<T> _M_kernel;

public:
    /// The kernel constructor
    div(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("div"))
    {}

    /// Invokes the kernel.
    ///
    /// \param input1 the divident
    /// \param input2 the divisor
    /// \returns a \ref future_tensor with the result.
    template <immutable_tensor_t<T> Input1, immutable_tensor_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2)
        requires(Input1::dim() > 1 && Input1::dim() == Input2::dim())
    {
        return _M_kernel(input1, input2);
    }

    /// Invokes the kernel by broadcasting the last dimension
    ///
    /// \param input1 the divident
    /// \param input2 the divisor
    /// \returns a \ref future_tensor with the result.
    template <immutable_tensor_t<T> Input1, immutable_tensor1_t<T> Input2>
    auto
    operator()(Input1 input1, Input2 input2) requires(Input1::dim() > 1)
    {
        auto div_size = input2.sizes().back();
        auto dim_size = input1.sizes().back();
        auto num_rows = input1.numel() / dim_size;

        if (num_rows != div_size) {
            throw std::runtime_error(std::format(
                "kernel::div: tensor sizes {} and {} are not broadcastable", num_rows, div_size
            ));
        }

        auto& accelerator = _M_kernel.get_accelerator();
        auto divisor = repeat_interleave(input2, dim_size, 0, accelerator);

        return operator()(input1, divisor.view(input1.shape()));
    }
};


} // namespace kernel
} // namespace metalchat
