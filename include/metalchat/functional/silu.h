#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class silu : public kernel {
private:
    inline static const std::string operation_name = "silu";

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

        auto threads = dim3(input.numel());
        auto thread = dim3(32);

        blocking(threads, thread)(n, input, output);
        return output;
    }
};


} // namespace metalchat
