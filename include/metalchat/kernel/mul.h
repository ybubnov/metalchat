#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class hadamard : public kernel {
private:
    inline static const std::string operation_name = "hadamard";

public:
    hadamard(device& device)
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <
        std::size_t M,
        std::size_t N,
        ContiguousContainer Input1Container,
        ContiguousContainer Input2Container>
    auto
    operator()(
        const tensor<T, M, Input1Container>& input1, const tensor<T, N, Input2Container>& input2
    )
    {
        constexpr std::size_t block_size = 32;

        auto data_size = input1.numel();
        auto dim_size = input1.sizes().back();
        auto num_rows = data_size / dim_size;

        if (auto dim_size2 = input2.sizes().back(); dim_size != dim_size2) {
            throw std::invalid_argument(std::format(
                "hadamard: last dimension size should be the same for both tensors {} != {}",
                dim_size, dim_size2
            ));
        }

        if (auto data_size2 = input2.numel(); data_size != data_size2) {
            throw std::invalid_argument(std::format(
                "hadamard: data size should be the same for both tensors {} != {}", data_size,
                data_size2
            ));
        }

        auto input1_numel = input1.numel();
        auto input2_numel = input2.numel();

        auto output = empty_like(input1, m_device);

        auto thread_size = ceil_div(dim_size, block_size);
        auto thread = dim3(thread_size);
        auto threads = dim3(thread_size * num_rows);

        blocking(threads, thread)(scalar<int32_t>(dim_size), input1, input2, output);
        return output;
    }
};


template <typename T> class scalar_mul : public kernel {
private:
    inline static const std::string operation_name = "scalar_mul";

public:
    scalar_mul(device& device)
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <
        std::size_t N,
        ContiguousContainer InputContainer,
        ContiguousContainer MultiplierContainer>
    auto
    operator()(
        const tensor<T, N, InputContainer>& input,
        const tensor<T, 0, MultiplierContainer>& multiplier
    )
    {
        auto output = empty_like(input, m_device);
        auto n = scalar<int32_t>(input.numel());

        auto threads = dim3(input.numel());
        auto thread = dim3(32);

        blocking(threads, thread)(n, input, multiplier, output);
        return output;
    }

    template <std::size_t N, ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, N, InputContainer>& input, const T multiplier)
    {
        return operator()(input, scalar(multiplier));
    }
};


} // namespace metalchat
