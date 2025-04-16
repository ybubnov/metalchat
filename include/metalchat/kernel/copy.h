#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_future.h>
#include <metalchat/tensor_shared.h>


namespace metalchat {


template <typename T> class cpy {
private:
    inline static const std::string operation_name = "copy";

    kernel_base _m_kernel;

    template <ContiguousContainer InputContainer, ContiguousContainer OutputContainer>
    auto
    copy(shared_tensor<T, 2, InputContainer> input, shared_tensor<T, 2, OutputContainer> output)
    {
        if (auto dim_size = output.sizes().back(); dim_size != input.sizes().back()) {
            throw std::invalid_argument(std::format(
                "kernel::copy: last dimension should be the same for both tensors {} != {}",
                input.sizes().back(), dim_size
            ));
        }

        if (auto data_size = output.numel(); data_size != input.numel()) {
            throw std::invalid_argument(std::format(
                "kernel::copy: data size should be the same for both tensors {} != {}",
                input.sizes().back(), data_size
            ));
        }

        constexpr std::size_t block_size = 32;
        auto [grid, thread] = make_kernel_grid_1d(input, block_size);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_front(output, input);

        return future_tensor(output, std::move(fn));
    }

public:
    cpy(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    /// Copy values from input to the output.
    ///
    /// The metal kernel implementation supports only copying of 2-dimensional tensors,
    /// considering that all dimensions that are larger than 1 (a vector) are simply batch
    /// dimensions, we could simply collapse all of them into a single batch dimension.
    ///
    /// The resulting tensor from the future operation is also 2-dimensional, therefore
    /// if caller wants to retain original dimensionality, she must keep the original
    /// output tensor.
    ///
    /// The operation is executed asynchronously on GPU, therefore output tensor should be
    /// allocated on GPU memory.
    template <std::size_t N, std::size_t M, ContiguousContainer InputContainer>
    auto
    operator()(shared_tensor<T, N, InputContainer> input, shared_tensor<T, M, device_ref<T>> output)
    {
        int input_dim = input.sizes().back();
        int output_dim = output.sizes().back();

        return copy(input.view({-1, input_dim}), output.view({-1, output_dim}));
    }
};


} // namespace metalchat
