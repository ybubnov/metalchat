#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class softmax : public kernel {
private:
    inline static const std::string operation_name = "softmax";

public:
    softmax(device& device)
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <std::size_t N, ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, N, InputContainer>& input)
    {
        auto n = scalar<int32_t>(input.numel());
        auto output = empty_like(input, m_device);

        auto groups = dim3(1);
        auto threads = dim3(ceil_div(input.numel(), 4));

        blocking(groups, threads)(n, input, output);
        return output;
    }
};


} // namespace metalchat
