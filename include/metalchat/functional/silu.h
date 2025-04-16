#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class silu : public kernel {
private:
    inline static const std::string operation_name = "silu";

    std::size_t
    ceil_div(std::size_t a, std::size_t b)
    {
        return (a + b - 1) / b;
    }

public:
    silu(device& device)
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <std::size_t N, ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, N, InputContainer>& input)
    {
        auto output = empty_like(input, m_device);
        auto n = scalar<int32_t>(input.numel());

        auto groups = dim3(ceil_div(input.numel(), 32));
        auto threads = dim3(32);

        blocking(groups, threads)(n, input, output);
        return output;
    }
};


} // namespace metalchat
