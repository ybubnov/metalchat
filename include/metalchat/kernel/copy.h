#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_future.h>
#include <metalchat/tensor_shared.h>


namespace metalchat {


template <typename T> class cpy : public base_kernel {
private:
    inline static const std::string operation_name = "copy";

    kernel_base _m_kernel;

public:
    cpy(device& device)
    : base_kernel(operation_name, type_traits<T>::name(), device),
      _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <ContiguousContainer InputContainer, ContiguousContainer OutputContainer>
    void
    copy(const tensor<T, 2, InputContainer>& input, const tensor<T, 2, OutputContainer>& output)
    {
        constexpr std::size_t block_size = 32;

        auto data_size = input.numel();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        if (auto dim_size2 = output.sizes().back(); dim_size != dim_size2) {
            throw std::invalid_argument(std::format(
                "kernel::cpy: last dimension should be the same for both tensors {} != {}",
                dim_size, dim_size2
            ));
        }

        if (auto data_size2 = output.numel(); data_size != data_size2) {
            throw std::invalid_argument(std::format(
                "kernel::cpy: data size should be the same for both tensors {} != {}", data_size,
                data_size2
            ));
        }

        auto thread_size = ceil_div(dim_size, block_size);
        auto thread = dim3(thread_size);
        auto threads = dim3(thread_size * num_rows);

        blocking(threads, thread)(scalar(output.layout()), output, scalar(input.layout()), input);
    }

    template <
        std::size_t N,
        std::size_t M,
        ContiguousContainer InputContainer,
        ContiguousContainer OutputContainer>
    void
    operator()(
        const tensor<T, N, InputContainer>& input, const tensor<T, M, OutputContainer>& output
    )
    {
        int input_dim = input.sizes().back();
        int output_dim = output.sizes().back();

        copy(input.view({-1, input_dim}), output.view({-1, output_dim}));
    }

    template <ContiguousContainer InputContainer, ContiguousContainer OutputContainer>
    auto
    copy(shared_tensor<T, 2, InputContainer> input, shared_tensor<T, 2, OutputContainer> output)
    {
        constexpr std::size_t block_size = 32;

        auto data_size = input.numel();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        if (auto dim_size2 = output.sizes().back(); dim_size != dim_size2) {
            throw std::invalid_argument(std::format(
                "kernel::cpy: last dimension should be the same for both tensors {} != {}",
                dim_size, dim_size2
            ));
        }

        if (auto data_size2 = output.numel(); data_size != data_size2) {
            throw std::invalid_argument(std::format(
                "kernel::cpy: data size should be the same for both tensors {} != {}", data_size,
                data_size2
            ));
        }

        auto thread_size = ceil_div(dim_size, block_size);
        auto thread = dim3(thread_size);
        auto grid = dim3(thread_size * num_rows);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_front(output, input);

        return future_tensor(output, std::move(fn));
    }

    template <
        std::size_t N,
        std::size_t M,
        ContiguousContainer InputContainer,
        ContiguousContainer OutputContainer>
    auto
    operator()(
        shared_tensor<T, N, InputContainer> input, shared_tensor<T, M, OutputContainer> output
    )
    {
        int input_dim = input.sizes().back();
        int output_dim = output.sizes().back();

        return copy(input.view({-1, input_dim}), output.view({-1, output_dim}));
    }
};


} // namespace metalchat
