#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class cpy : public base_kernel {
private:
    inline static const std::string operation_name = "copy";

public:
    cpy(device& device)
    : base_kernel(operation_name, type_traits<T>::name(), device)
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

        blocking(threads, thread)(scalar(input.layout()), scalar(output.layout()), input, output);
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

    template <
        std::size_t N,
        std::size_t M,
        ContiguousContainer InputContainer,
        ContiguousContainer OutputContainer>
    auto
    maybe_compute(
        const tensor<T, N, InputContainer>& input, const tensor<T, M, OutputContainer>& output
    )
    {
        int input_dim = input.sizes().back();
        int output_dim = output.sizes().back();

        auto input_view = input.view({-1, input_dim});
        auto output_view = output.view({-1, output_dim});

        constexpr std::size_t block_size = 32;

        auto data_size = input_view.numel();
        auto dim_size = input_view.sizes().back();
        auto num_rows = data_size / dim_size;

        auto thread_size = ceil_div(dim_size, block_size);
        auto thread = dim3(thread_size);
        auto threads = dim3(thread_size * num_rows);

        auto cb = unsequenced_policy_callback<T, M>{
            .output = nullptr,
            .promise = std::make_shared<std::promise<bool>>()
        };

        auto policy = nonblocking(threads, thread, std::move(cb));
        policy(scalar(input_view.layout()), scalar(output_view.layout()), input_view, output_view);

        return policy;
    }
};


} // namespace metalchat
