// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025-2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/expected.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 8> class matmul {
private:
    basic_kernel _M_gemm;
    basic_kernel _M_gemv;

public:
    matmul(hardware_accelerator& gpu)
    : _M_gemm(gpu.load<T>("gemm", BlockSize)),
      _M_gemv(gpu.load<T>("gemv"))
    {}

    template <immutable_tensor3_t<T> Input, immutable_tensor3_t<T> Weight>
    auto
    operator()(Input input, Weight weight)
    {
        auto num_batches = input.size(0);
        auto input_size1 = input.size(1);
        auto dim_size = input.size(2);
        auto weight_size2 = weight.size(2);

        // Batched matmul does not support broadcasting operations, therefore throw an
        // exception, when the number of batches for input tensors are different.
        auto expected_input = expected_tensor(input)
                                  .expect(matching_dim(0, weight.size(0)))
                                  .expect(matching_dim(2, weight.size(1)));

        auto alloc = _M_gemv.get_allocator();
        auto output = shared_empty<T>({num_batches, input_size1, weight_size2}, alloc);

        if (input_size1 == 1) {
            auto max_threads = _M_gemv.max_threads_per_threadgroup();
            auto block_size = ceil_div(dim_size, max_threads);
            auto thread_size = ceil_div(dim_size, block_size);

            auto thread = dim3(thread_size);
            auto grid = dim3(thread_size * weight_size2, num_batches);

            auto task = kernel_task(_M_gemv, grid, thread);
            auto task_future = task.bind_front(
                flatten<2>(output), flatten<2>(expected_input), weight, scalar<int32_t>(block_size)
            );

            return future_tensor(output, std::move(task_future));
        }

        auto grid = dim3(
            ceil_div(input_size1, BlockSize) * BlockSize,
            ceil_div(weight_size2, BlockSize) * BlockSize, num_batches
        );
        auto thread = dim3(BlockSize, BlockSize);

        auto task = kernel_task(_M_gemm, grid, thread);
        auto task_future = task.bind_front(output, expected_input, weight);

        // A(MxK) @ B(KxN) -> C(MxN)
        return future_tensor(output, std::move(task_future));
    }

    template <immutable_tensor3_t<T> Input, immutable_tensor2_t<T> Weight>
    auto
    operator()(Input input, Weight weight)
    {
        // TODO: does it make sense to call repeat_interleave for the number of batches > 1?
        return operator()(input, weight.expand_dims(0));
    }

    template <immutable_tensor2_t<T> Input, immutable_tensor2_t<T> Weight>
    auto
    operator()(Input input, Weight weight)
    {
        auto output = operator()(input.expand_dims(0), weight.expand_dims(0));

        int input_size0 = input.size(0);
        int weight_size1 = weight.size(1);
        return output.view({input_size0, weight_size1});
    }

    template <immutable_tensor_t<T> Input, immutable_tensor_t<T> Weight>
    auto
    operator()(Input input, Weight weight)
        requires(Input::dim() == Weight::dim() && Input::dim() > 3)
    {
        constexpr std::size_t N = Input::dim();

        int output_sizes[N];
        std::copy(input.sizes().begin(), input.sizes().end(), output_sizes);
        output_sizes[N - 2] = input.size(N - 2);
        output_sizes[N - 1] = weight.size(N - 1);

        auto output = operator()(flatten<3>(input), flatten<3>(weight));
        return output.view(std::move(output_sizes));
    }
};


} // namespace kernel
} // namespace metalchat
