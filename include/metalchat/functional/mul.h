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

    template <ContiguousContainer Input1Container, ContiguousContainer Input2Container>
    auto
    operator()(
        const tensor<T, 2, Input1Container>& input1, const tensor<T, 2, Input2Container>& input2
    )
    {
        assert(input1.size(0) == input2.size(0));
        assert(input1.size(1) == input2.size(1));

        auto output = empty<T>({input1.size(0), input1.size(1)}, m_device);

        auto m = scalar<int32_t>(input1.size(0));
        auto n = scalar<int32_t>(input1.size(1));

        auto groups = dim3(ceil_div(input1.size(0), 32), ceil_div(input1.size(1), 32));
        auto threads = dim3(32, 32, 1);

        blocking(groups, threads)(m, n, input1, input2, output);
        return output;
    }
};


} // namespace metalchat
