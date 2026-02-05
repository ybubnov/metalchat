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


template <typename T, std::size_t BlockSize = 8> class add2 {
private:
    basic_kernel _M_kernel;

public:
    add2(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("add2", BlockSize))
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

        auto expected_input1 =
            expected_tensor(input1).same_dim(input2, M - 2, 0).same_dim(input2, M - 1, 1).value();

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
/// ```c++
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
