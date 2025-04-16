#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class cpy : public kernel {
private:
    inline static const std::string operation_name = "copy";

public:
    cpy(device& device)
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <
        std::size_t N,
        ContiguousContainer InputContainer,
        ContiguousContainer OutputContainer>
    auto
    copy(
        const tensor_base<T, N, InputContainer>& input,
        const tensor_base<T, N, OutputContainer>& output
    )
    {
        constexpr std::size_t block_size = 32;

        auto data_size = input.numel();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        if (auto dim_size2 = output.sizes().back(); dim_size != dim_size2) {
            throw std::invalid_argument(std::format(
                "copy: last dimension should be the same for both tensors {} != {}", dim_size,
                dim_size2
            ));
        }

        if (auto data_size2 = output.numel(); data_size != data_size2) {
            throw std::invalid_argument(std::format(
                "copy: data size should be the same for both tensors {} != {}", data_size,
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
    auto
    operator()(
        const tensor_base<T, N, InputContainer>& input,
        const tensor_base<T, M, OutputContainer>& output
    )
    {
        int input_dim = input.sizes().back();
        int output_dim = output.sizes().back();

        return copy(input.view({-1, input_dim}), output.view({-1, output_dim}));
    }
};


} // namespace metalchat
