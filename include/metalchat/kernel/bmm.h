#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T, std::size_t BlockSize = 16> class bmm {
private:
    inline static const std::string operation_name = "bmm_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    bmm(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor3d InputTensor, immutable_tensor3d WeightTensor>
    auto
    operator()(InputTensor input, WeightTensor weight)
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

        auto grid = dim3(
            ceil_div(input_size1, BlockSize) * BlockSize,
            ceil_div(weight_size2, BlockSize) * BlockSize, num_batches
        );
        auto thread = dim3(BlockSize, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input, weight);

        // A(MxK) @ B(KxN) -> C(MxN)
        return empty_future<T>({num_batches, input_size1, weight_size2}, std::move(fn));
    }

    template <immutable_tensor3d InputTensor, immutable_tensor2d WeightTensor>
    auto
    operator()(InputTensor input, WeightTensor weight)
    {
        // TODO: does it make sense to call repeat_interleave for the number of batches > 1?
        return operator()(input, weight.expand_dims(0));
    }

    template <immutable_tensor2d InputTensor, immutable_tensor2d WeightTensor>
    auto
    operator()(InputTensor input, WeightTensor weight)
    {
        auto output = operator()(input.expand_dims(0), weight.expand_dims(0));

        int input_size0 = input.size(0);
        int weight_size1 = weight.size(1);
        return output.view({input_size0, weight_size1});
    }

    template <immutable_tensor InputTensor, immutable_tensor WeightTensor>
    auto
    operator()(InputTensor input, WeightTensor weight)
        requires(InputTensor::dim() == WeightTensor::dim() && InputTensor::dim() > 3)
    {
        constexpr std::size_t N = InputTensor::dim();

        int input_size0 = input.size(N - 2);
        int weight_size1 = weight.size(N - 1);

        int output_sizes[N];
        std::copy(input.sizes().begin(), input.sizes().end(), output_sizes);
        output_sizes[N - 2] = input.size(N - 2);
        output_sizes[N - 1] = weight.size(N - 1);

        auto output = operator()(input.template flatten<3>(), weight.template flatten<3>());
        return output.view(std::move(output_sizes));
    }
};


} // namespace metalchat
