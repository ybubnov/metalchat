#pragma once

#include <format>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class embedding : public kernel {
private:
    inline static const std::string operation_name = "embedding";

public:
    embedding(device& device)
    : kernel(std::format("{}_{}", operation_name, type_traits<T>::name()), device)
    {}

    template <ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<int32_t, 1, InputContainer>& input, const tensor<T, 2, WeightContainer>& weight
    )
    {
        auto stride = scalar<int32_t>(weight.stride(0));
        auto output = empty<T>({input.size(0), weight.size(1)}, m_device);

        auto blocks = dim3(input.size(0), weight.size(1));
        blocking(blocks, 1)(input, weight, stride, output);

        return output;
    }
};


} // namespace metalchat
