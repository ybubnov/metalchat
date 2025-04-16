#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 8> class bmm {
private:
    inline static const std::string operation_name = "bmm_" + std::to_string(BlockSize);

    basic_kernel _m_kernel;

public:
    bmm(hardware_accelerator& gpu)
    : _m_kernel(gpu.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor3_t<T> Input, immutable_tensor3_t<T> Weight>
    auto
    operator()(Input input, Weight weight)
    {
        auto num_batches = input.size(0);
        auto input_size1 = input.size(1);
        auto weight_size2 = weight.size(2);

        // Batched matmul does not support broadcasting operations, therefore throw an
        // exception, when the number of batches for input tensors are different.
        if (auto num_weight_batches = weight.size(0); num_batches != num_weight_batches) {
            throw std::invalid_argument(std::format(
                "kernel::bmm: batches of the input tensors should be the same {} != {}",
                num_batches, num_weight_batches
            ));
        }

        if (input.size(2) != weight.size(1)) {
            throw std::invalid_argument(std::format(
                "kernel::bmm: matrices are with different inner dimension ({}x{}) and ({}x{})",
                input.size(1), input.size(2), weight.size(1), weight.size(2)
            ));
        }

        auto output
            = shared_empty<T>({num_batches, input_size1, weight_size2}, _m_kernel.get_allocator());

        constexpr std::size_t BM = 64;
        constexpr std::size_t BN = 64;
        constexpr std::size_t BK = 8;
        constexpr std::size_t TM = 8;

        // auto grid = dim3(
        //     ceil_div(input_size1, BlockSize) * BlockSize,
        //     ceil_div(weight_size2, BlockSize) * BlockSize, num_batches
        //);
        // auto thread = dim3(BlockSize, BlockSize);
        auto block_size = (BM * BN) / TM;
        auto N = weight.size(2);
        auto M = input.size(1);

        auto grid = dim3(ceil_div(N, BN) * block_size, ceil_div(M, BM), num_batches);
        auto thread = dim3(block_size);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(output, input, weight);

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
