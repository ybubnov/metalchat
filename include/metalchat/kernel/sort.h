// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


/// Returns the indices of the buckets to which each value in the input belongs, where the
/// boundaries of the buckets are set by boundaries. Return a new tensor with the same size
/// as input. If right is false (default), then the left boundary is open.
///
/// \tparam T A type of the input data.
template <typename T> class bucketize {
private:
    basic_kernel _M_kernel;

public:
    /// The \ref bucketize kernel constructor.
    bucketize(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("bucketize"))
    {}

    /// Invokes the kernel.
    ///
    /// \param input 2-D tensor containing search values.
    /// \param boundaries 1-D tensor must contain a strictly increasing sequence, or the return
    ///     value is undefined.
    /// \param right Determines the side of the interval to return.
    ///
    /// \return a \ref future_tensor with the indices of the boundaries.
    template <immutable_tensor2_t<T> Input, immutable_tensor1_t<T> Boundaries>
    auto
    operator()(Input input, Boundaries boundaries, bool right = false)
    {
        auto num_rows = input.size(0);
        auto dim_size = input.size(1);
        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto [grid, thread] = make_kernel_grid_2d(num_rows, dim_size, max_threads);

        auto alloc = _M_kernel.get_allocator();
        auto output = shared_empty<int32_t>({num_rows, dim_size}, alloc);

        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(output, input, boundaries, scalar<bool>(right));

        return future_tensor(output, std::move(task_future));
    }

    /// Invokes the kernel.
    ///
    /// \param input N-D tensor container search values.
    /// \param boundaries 1-D tensor specifying interval boundaries.
    /// \param right Determines the side of the interval to return.
    ///
    /// \return a \ref future_tensor with the indices of the boundaries.
    template <immutable_tensor_t<T> Input, immutable_tensor1_t<T> Boundaries>
    auto
    operator()(Input input, Boundaries boundaries, bool right = false)
    {
        auto input_view = flatten<2>(input);
        auto output = operator()(input_view, boundaries, right);
        return output.view(input.shape());
    }
};


template <typename T> class sort {
private:
    basic_kernel _M_kernel;

public:
    sort(hardware_accelerator& gpu)
    : _M_kernel(gpu.load<T>("sort"))
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        auto dim_size = input.sizes().back();
        auto num_rows = input.numel() / dim_size;

        auto input_view = input.view({-1, int(dim_size)});
        auto dim_size_aligned = ceil_pow2(dim_size);

        auto alloc = _M_kernel.get_allocator();
        auto values = shared_empty<T>({num_rows, dim_size_aligned}, alloc);
        auto indices = shared_empty<int32_t>({num_rows, dim_size_aligned}, alloc);

        auto max_threads = _M_kernel.max_threads_per_threadgroup();
        auto block_size = ceil_div(dim_size_aligned, max_threads);
        auto thread_size = ceil_div(dim_size_aligned, block_size);

        auto thread = dim3(thread_size);
        auto grid = dim3(thread_size * num_rows);

        auto block_tensor = scalar<uint32_t>(block_size);
        auto task = kernel_task(_M_kernel, grid, thread);
        auto task_future = task.bind_front(values, indices, input_view, block_tensor);

        // A single kernel task produces both outputs (values and indices), but a future
        // tensor can hold only a single output. To work this around, we return to future
        // tensors, one depending on another.
        auto values_future = future_tensor(values, std::move(task_future));
        auto indices_future = future_tensor(indices, values_future);

        // The output dimension size is scaled to a power of 2, but the input tensor might
        // be a different size. Slice the result according to the input dimension size, and
        // then rescale batch dimensions as they where originally defined in the input
        // tensor.
        auto values_sorted = values_future[slice(), slice(0, dim_size)].view(input.shape());
        auto indices_sorted = indices_future[slice(), slice(0, dim_size)].view(input.shape());

        return std::make_tuple(values_sorted, indices_sorted);
    }
};


} // namespace kernel
} // namespace metalchat
