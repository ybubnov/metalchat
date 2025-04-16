#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class mul : public kernel {
private:
    inline static const std::string operation_name = "mul";

    std::size_t
    ceil_div(std::size_t a, std::size_t b)
    {
        return (a + b - 1) / b;
    }

public:
    mul(device& device)
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
        assert(input1.numel() == input2.numel());

        auto output = empty_like(input1, m_device);
        auto n = scalar<int32_t>(input1.numel());

        auto groups = dim3(ceil_div(input1.numel(), 32));
        auto threads = dim3(32);

        blocking(groups, threads)(n, input1, input2, output);
        return output;
    }
};


} // namespace metalchat
