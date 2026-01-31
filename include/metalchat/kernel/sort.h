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


inline std::size_t
__ceil_pow2(std::size_t value)
{
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value++;
    return value;
}


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
        auto dim_size_aligned = __ceil_pow2(dim_size);

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
