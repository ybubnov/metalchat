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


/// Create a copy of a tensor.
///
/// The metal kernel implementation supports only copying of 2-dimensional tensors,
/// considering that all dimensions that are larger than 1 (a vector) are simply batch
/// dimensions, we could simply collapse all of them into a single batch dimension.
///
/// The resulting tensor from the future operation is also 2-dimensional, therefore
/// if caller wants to retain original dimensionality, she must keep the original
/// output tensor or adjust the resulting tensor shape as needed.
///
/// \note The operation is executed asynchronously on GPU, therefore output tensor should
/// be allocated on GPU memory beforehand.
template <typename T> class clone {
private:
    basic_kernel _M_kernel;

    template <immutable_tensor2_t<T> Input, immutable_tensor2_t<T> Output>
    auto
    copy(Input input, Output output)
    {
        auto expected_input =
            expected_tensor(input).same_last_dim(output).same_numel(output).value();

        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto [grid, thread] = make_dynamic_kernel_grid_2d(expected_input, max_threads);

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output, expected_input);

        return future_tensor(output, std::move(task_future));
    }

public:
    /// The kernel constructor.
    clone(hardware_accelerator& accelerator)
    : _M_kernel(accelerator.load<T>("copy"))
    {}

    /// Invokes the kernel.
    ///
    /// \param input a tensor to clone data from.
    /// \param output a tensor to clone data to.
    ///
    /// \return a \ref future_tensor with the data copied from an input tensor.
    template <immutable_tensor_t<T> Input, immutable_hardware_tensor_t<T> Output>
    auto
    operator()(Input input, Output output)
    {
        return copy(flatten<2>(input), flatten<2>(output));
    }

    /// Creates an output tensor like the input and invokes the kernel.
    ///
    /// \param input a tensor to clone the data from.
    ///
    /// \return a \ref future_tensor with the data copied from an input tensor.
    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        auto output = shared_empty_like<T>(input, _M_kernel.get_allocator());
        return operator()(input, output);
    }
};


/// Writes values into the tensor at the specified indices.
///
/// \warning When indices are not unique, the behaviour is non-deterministic.
template <typename T> class scatter {
private:
    basic_kernel _M_kernel;

public:
    /// The kernel constructor.
    scatter(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("scatter"))
    {}

    /// Invokes the kernel, and writes a single value to the output tensor according to
    /// the specified boolean mask.
    ///
    /// \param output a tensor to write data to.
    /// \param mask a boolean mask tensor (should be the same size as an output tensor).
    /// \param value a value to write.
    ///
    /// \return a \ref future_tensor with the kernel operation result.
    ///
    /// ```c++
    /// auto T = tensor<float>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    /// auto M = tensor<bool>({{true, false, false}, {false, true, true}});
    ///
    /// auto accelerator = hardware_accelerator();
    /// auto scatter = kernel::scatter(accelerator);
    ///
    /// auto output = scatter(T, M, 9.0);
    /// std::cout << output.get() << std::endl;
    /// // out:
    /// // [[9.0, 2.0, 3.0],
    /// //  [4.0, 9.0, 9.0]], sizes=(2, 3)
    /// ```
    template <immutable_tensor_t<T> Output, immutable_tensor_t<bool> Mask>
    auto
    operator()(Output output, Mask mask, T value)
    {
        auto expected_output = expected_tensor(output).same_shape(mask).value();

        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto [grid, thread] = make_dynamic_kernel_grid_2d(expected_output, max_threads);

        auto output_view = flatten<2>(expected_output);
        auto mask_view = flatten<2>(mask);

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, mask_view, scalar(value));

        auto result = future_tensor(expected_output, std::move(task_future));
        return result.view(expected_output.shape());
    }
};


/// Gathers values given the index tensor.
///
/// ```c++
/// auto T = tensor<float>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
/// auto index = tensor<int32_t>({{0, 0}, {1, 0}});
///
/// auto accelerator = hardware_accelerator();
/// auto gather = kernel::gather(accelerator);
///
/// auto output = gather(T, index);
/// std::cout << output.get() << std::endl;
/// // out:
/// // [[1.0, 1.0],
/// //  [5.0, 4.0]], sizes=(2, 2)
/// ```
///
/// \note Current implementation treats all tensors as 2-dimensional with dimension 0 as a batch
/// dimension, and gather elements only along 0 dimension.
template <typename T> class gather {
private:
    basic_kernel _M_kernel;

public:
    /// The kernel constructor.
    gather(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("gather"))
    {}

    /// Invokes the kernel.
    ///
    /// \param input an input tensor to gather values from.
    /// \param index an index tensor that specifies locations of elements within input tensor.
    ///
    /// \return a \ref future_tensor with the elements gathered from an input tensor.
    template <immutable_tensor_t<T> Input, immutable_tensor_t<int32_t> Index>
    auto
    operator()(Input input, Index index)
    {
        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto [grid, thread] = make_dynamic_kernel_grid_2d(index, max_threads);

        auto input_view = flatten<2>(input);
        auto index_view = flatten<2>(index);
        auto output_view = shared_empty_like<T>(index_view, _M_kernel.get_allocator());

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output_view, input_view, index_view);

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(index.shape());
    }
};


} // namespace kernel
} // namespace metalchat
